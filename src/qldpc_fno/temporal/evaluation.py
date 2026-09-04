"""Paired, sequence-preserving evaluation for causal decoder priors.

Gate-2 helpers in this module deliberately report descriptive quantities only.
They do not perform model selection on decoder outcomes and do not attach a
hypothesis decision to the reduced screen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse

from qldpc_fno.decoders.bplsd import BPLSDConfig, decode_bplsd_prior_batch
from qldpc_fno.temporal.baselines import (
    CircularLogisticForecaster,
    StationaryForecaster,
    calibrate_temperature,
    fit_ewma,
    fit_logistic_ar,
    fit_stationary,
)


@dataclass(frozen=True, slots=True)
class ForecastSplit:
    """Separated observed/target arrays for one fitting role."""

    role: str
    sequence_ids: tuple[str, ...]
    syndromes: np.ndarray
    targets: np.ndarray
    scored_mask: np.ndarray

    def __post_init__(self) -> None:
        syndromes = np.asarray(self.syndromes)
        targets = np.asarray(self.targets)
        mask = np.asarray(self.scored_mask)
        if self.role not in {"train", "validation", "calibration"}:
            raise ValueError("forecast split role must be train, validation, or calibration")
        if syndromes.ndim != 4 or targets.ndim != 4:
            raise ValueError("forecast arrays must have shape (sequences, rounds, channels, ell)")
        if syndromes.shape[:2] != targets.shape[:2] or syndromes.shape[-1] != targets.shape[-1]:
            raise ValueError("forecast observations and targets must share sequence geometry")
        _validate_sequence_ids(self.sequence_ids, syndromes.shape[0])
        if mask.shape == (syndromes.shape[1],):
            mask = np.broadcast_to(mask, syndromes.shape[:2]).copy()
            object.__setattr__(self, "scored_mask", mask)
        elif mask.shape != syndromes.shape[:2]:
            raise ValueError("scored_mask must select rounds per sequence")
        if mask.dtype != np.bool_ or not mask.any():
            raise ValueError("scored_mask must be Boolean and nonempty")
        if not np.all(mask.any(axis=1)):
            raise ValueError("every complete sequence must contain a scored round")


@dataclass(frozen=True, slots=True)
class CausalEvaluationBatch:
    """Complete paired validation sequences, including evaluation-only labels."""

    regime: str
    role: str
    sequence_ids: tuple[str, ...]
    syndromes: np.ndarray
    errors: np.ndarray
    logical_flips: np.ndarray
    scored_mask: np.ndarray

    def __post_init__(self) -> None:
        syndromes = np.asarray(self.syndromes)
        errors = np.asarray(self.errors)
        logical = np.asarray(self.logical_flips)
        mask = np.asarray(self.scored_mask)
        if self.role not in {"validation", "test"}:
            raise ValueError("causal evaluation role must be validation or test")
        if syndromes.ndim not in {3, 4} or errors.ndim not in {3, 4}:
            raise ValueError("syndromes and errors must contain sequence and round axes")
        if logical.ndim != 3:
            raise ValueError("logical_flips must have shape (sequences, rounds, logicals)")
        if syndromes.shape[:2] != errors.shape[:2] or syndromes.shape[:2] != logical.shape[:2]:
            raise ValueError("all evaluation arrays must share sequence and round dimensions")
        _validate_sequence_ids(self.sequence_ids, syndromes.shape[0])
        if mask.shape == (syndromes.shape[1],):
            mask = np.broadcast_to(mask, syndromes.shape[:2]).copy()
            object.__setattr__(self, "scored_mask", mask)
        elif mask.shape != syndromes.shape[:2]:
            raise ValueError("scored_mask must select rounds per sequence")
        if mask.dtype != np.bool_ or not mask.any():
            raise ValueError("scored_mask must be Boolean and nonempty")
        if not np.all(mask.any(axis=1)):
            raise ValueError("every complete sequence must contain a scored round")
        for name, values in (
            ("syndromes", syndromes),
            ("errors", errors),
            ("logical_flips", logical),
        ):
            if not np.all((values == 0) | (values == 1)):
                raise ValueError(f"{name} must be binary")


@dataclass(frozen=True, slots=True)
class BaselineSelection:
    models: dict[str, object]
    validation_nll: dict[str, float]
    selected_name: str
    frozen: bool = True


@dataclass(frozen=True, slots=True)
class ArmEvaluation:
    name: str
    sequence_membership: tuple[tuple[str, int], ...]
    per_sequence_nll: np.ndarray
    per_sequence_brier: np.ndarray
    per_sequence_bler: np.ndarray
    overall_nll: float
    overall_brier: float
    overall_bler: float
    convergence_rate: float
    mean_iterations: float
    mean_correction_weight: float
    latency_p50_seconds: float
    latency_p95_seconds: float
    latency_p99_seconds: float
    expected_calibration_error: float
    reliability: tuple[dict[str, float | int], ...]
    logical_failures: np.ndarray
    syndrome_valid: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray
    correction_weights: np.ndarray
    setup_latency_seconds: np.ndarray
    decode_latency_seconds: np.ndarray
    stored_parameters: int
    effective_parameters: int
    per_round_outcomes: tuple[dict[str, object], ...]

    @property
    def all_syndrome_valid(self) -> bool:
        return bool(np.all(self.syndrome_valid))


@dataclass(frozen=True, slots=True)
class PairedCausalEvaluation:
    regime: str
    role: str
    sequence_membership: tuple[tuple[str, int], ...]
    arms: dict[str, ArmEvaluation]


@dataclass(frozen=True, slots=True)
class ReducedProgression:
    regime: str
    selected_name: str
    stationary_nll: tuple[float, ...]
    selected_nll: tuple[float, ...]
    stationary_bler: tuple[float, ...]
    selected_bler: tuple[float, ...]
    nll_improvement: float
    bler_difference: float
    progressed: bool
    scope: str = "descriptive_reduced_non_scientific"
    p_value: None = None
    hypothesis_status: None = None


@dataclass(frozen=True, slots=True)
class ReducedFactorDiagnostics:
    in_basis_interaction: np.ndarray
    basis_mismatch_interaction: np.ndarray
    in_basis_mean: float
    basis_mismatch_mean: float
    same_direction: bool
    scope: str = "descriptive_reduced_non_scientific"
    p_value: None = None
    hypothesis_status: None = None


def _as_flat(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    return array.reshape(*array.shape[:2], -1)


def _validate_sequence_ids(sequence_ids: tuple[str, ...], sequence_count: int) -> None:
    if len(sequence_ids) != sequence_count or len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError("sequence_ids must uniquely identify every complete sequence")
    if any(re.fullmatch(r"[0-9a-f]{64}", identity) is None for identity in sequence_ids):
        raise ValueError("sequence_ids must be lowercase SHA-256 provenance identities")


def _forecast_nll_by_sequence(
    probabilities: np.ndarray, targets: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    p = np.clip(_as_flat(probabilities).astype(np.float64), 1e-12, 1.0 - 1e-12)
    y = _as_flat(targets).astype(np.float64)
    if p.shape != y.shape:
        raise ValueError("forecast probabilities and physical error targets must have equal shapes")
    losses = -(y * np.log(p) + (1.0 - y) * np.log1p(-p))
    return np.asarray([losses[index, mask[index]].mean() for index in range(p.shape[0])])


def _forecast_brier_by_sequence(
    probabilities: np.ndarray, targets: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    p = _as_flat(probabilities).astype(np.float64)
    y = _as_flat(targets).astype(np.float64)
    if p.shape != y.shape:
        raise ValueError("forecast probabilities and physical error targets must have equal shapes")
    losses = (p - y) ** 2
    return np.asarray([losses[index, mask[index]].mean() for index in range(p.shape[0])])


def _reliability_table(
    probabilities: np.ndarray, targets: np.ndarray, mask: np.ndarray, *, bins: int = 10
) -> tuple[float, tuple[dict[str, float | int], ...]]:
    selected = np.repeat(mask[:, :, None], _as_flat(targets).shape[-1], axis=2)
    p = _as_flat(probabilities)[selected].astype(np.float64)
    y = _as_flat(targets)[selected].astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.minimum(np.searchsorted(edges, p, side="right") - 1, bins - 1)
    rows: list[dict[str, float | int]] = []
    weighted_gap = 0.0
    for index in range(bins):
        chosen = indices == index
        count = int(chosen.sum())
        confidence = float(p[chosen].mean()) if count else 0.0
        frequency = float(y[chosen].mean()) if count else 0.0
        weighted_gap += count * abs(confidence - frequency)
        rows.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_probability": confidence,
                "empirical_frequency": frequency,
            }
        )
    return weighted_gap / p.size, tuple(rows)


def _predict(model: object, split: ForecastSplit) -> np.ndarray:
    if isinstance(model, StationaryForecaster):
        return model.predict(
            sequence_count=split.syndromes.shape[0], rounds=split.syndromes.shape[1]
        )
    return np.asarray(model.predict(split.syndromes))  # type: ignore[attr-defined]


def _raw_predict(model: object, split: ForecastSplit) -> np.ndarray:
    if isinstance(model, StationaryForecaster):
        return model.raw_predict(
            sequence_count=split.syndromes.shape[0], rounds=split.syndromes.shape[1]
        )
    return np.asarray(model.raw_predict(split.syndromes))  # type: ignore[attr-defined]


def fit_select_observation_baselines(
    *,
    train: ForecastSplit,
    validation: ForecastSplit,
    calibration: ForecastSplit,
    max_iter: int = 500,
) -> BaselineSelection:
    """Fit train-only weights, tune on validation NLL, calibrate on calibration."""

    if (train.role, validation.role, calibration.role) != (
        "train",
        "validation",
        "calibration",
    ):
        raise ValueError("baseline fitting requires train, validation, calibration in that order")
    identities = (
        set(train.sequence_ids),
        set(validation.sequence_ids),
        set(calibration.sequence_ids),
    )
    if any(identities[left] & identities[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("train, validation, and calibration sequence identities must be disjoint")
    stationary = fit_stationary(
        train.targets,
        validation.targets,
        train_mask=train.scored_mask,
        validation_mask=validation.scored_mask,
        calibrate=None,
    )
    ewma = fit_ewma(
        train.syndromes,
        train.targets,
        validation.syndromes,
        validation.targets,
        train_mask=train.scored_mask,
        validation_mask=validation.scored_mask,
        max_iter=max_iter,
        calibrate=None,
    )
    logistic = fit_logistic_ar(
        train.syndromes,
        train.targets,
        validation.syndromes,
        validation.targets,
        train_mask=train.scored_mask,
        validation_mask=validation.scored_mask,
        max_iter=max_iter,
        calibrate=None,
    )
    uncalibrated: dict[str, object] = {
        "stationary_field": stationary,
        "ewma": ewma,
        "logistic_ar": logistic,
    }
    scores = {
        name: float(
            _forecast_nll_by_sequence(
                _predict(model, validation), validation.targets, validation.scored_mask
            ).mean()
        )
        for name, model in uncalibrated.items()
    }
    selected = "ewma" if scores["ewma"] <= scores["logistic_ar"] else "logistic_ar"
    models: dict[str, object] = {}
    for name, model in uncalibrated.items():
        raw_calibration = _raw_predict(model, calibration)
        temperature = calibrate_temperature(
            raw_calibration,
            calibration.targets,
            calibration.scored_mask,
        )
        models[name] = replace(model, temperature=temperature)
    return BaselineSelection(models, scores, selected)


def baseline_parameter_accounting(model: object) -> tuple[int, int]:
    """Count stored scalar values for fitted, non-neural forecasters."""

    if isinstance(model, StationaryForecaster):
        stored = int(model.empirical_field.size + 3)
    elif isinstance(model, CircularLogisticForecaster):
        stored = int(model.weight.size + model.bias.size + 3)
    else:
        stored = sum(
            int(np.asarray(value).size)
            for value in vars(model).values()
            if isinstance(value, np.ndarray)
        )
    return stored, stored


def evaluate_causal_arms(
    batch: CausalEvaluationBatch,
    arm_priors: dict[str, np.ndarray],
    *,
    hx: sparse.spmatrix,
    logical_x: sparse.spmatrix,
    parameter_counts: dict[str, tuple[int, int]],
    decoder_config: BPLSDConfig | None = None,
) -> PairedCausalEvaluation:
    """Decode every prior arm on the exact same scored sequence-rounds."""

    if not arm_priors:
        raise ValueError("at least one deployable arm is required")
    if any("oracle" in name.lower() for name in arm_priors):
        raise ValueError("privileged oracle is generator sanity only, not a decoder arm")
    if set(parameter_counts) != set(arm_priors):
        raise ValueError("parameter accounting must be supplied for every and only decoder arm")
    if any(
        type(stored) is not int
        or type(effective) is not int
        or stored < 0
        or effective < 0
        or effective > stored
        for stored, effective in parameter_counts.values()
    ):
        raise ValueError("parameter counts must satisfy 0 <= effective <= stored")
    pinned_decoder_config = BPLSDConfig() if decoder_config is None else decoder_config
    syndromes = _as_flat(batch.syndromes)
    errors = _as_flat(batch.errors)
    logical = np.asarray(batch.logical_flips)
    if syndromes.shape[-1] != hx.shape[0] or errors.shape[-1] != hx.shape[1]:
        raise ValueError("evaluation arrays do not match parity-check geometry")
    if logical.shape[-1] != logical_x.shape[0] or logical_x.shape[1] != hx.shape[1]:
        raise ValueError("logical labels or operators do not match code geometry")
    selected_syndromes = syndromes[batch.scored_mask]
    selected_logical = logical[batch.scored_mask]
    membership = tuple(
        (identity, round_index)
        for sequence_index, identity in enumerate(batch.sequence_ids)
        for round_index in np.flatnonzero(batch.scored_mask[sequence_index])
    )
    arms: dict[str, ArmEvaluation] = {}
    for name, probabilities in arm_priors.items():
        priors = _as_flat(probabilities).astype(np.float64)
        if priors.shape != errors.shape:
            raise ValueError(f"arm {name!r} priors do not match evaluation error geometry")
        if not np.all(np.isfinite(priors)) or not np.all((priors > 0.0) & (priors < 0.5)):
            raise ValueError(f"arm {name!r} priors must be finite and strictly between 0 and 0.5")
        decoded = decode_bplsd_prior_batch(
            hx,
            selected_syndromes,
            logical_x,
            error_channels=priors[batch.scored_mask],
            config=pinned_decoder_config,
        )
        recomputed_valid = np.all(
            (np.asarray(hx.tocsr() @ decoded.corrections.T).T % 2) == selected_syndromes,
            axis=1,
        )
        if not np.array_equal(recomputed_valid, decoded.syndrome_valid):
            raise RuntimeError(
                "decoder syndrome_valid disagrees with independently recomputed validity"
            )
        logical_mismatch = np.any(decoded.predicted_observables != selected_logical, axis=1)
        failures = (~recomputed_valid) | logical_mismatch
        selected_priors = priors[batch.scored_mask]
        selected_errors = errors[batch.scored_mask].astype(np.float64)
        round_nll = -(
            selected_errors * np.log(selected_priors)
            + (1.0 - selected_errors) * np.log1p(-selected_priors)
        ).mean(axis=1)
        round_brier = ((selected_priors - selected_errors) ** 2).mean(axis=1)
        per_sequence_nll = _forecast_nll_by_sequence(priors, errors, batch.scored_mask)
        per_sequence_brier = _forecast_brier_by_sequence(priors, errors, batch.scored_mask)
        per_sequence_bler = np.asarray(
            [
                failures[
                    [index for index, member in enumerate(membership) if member[0] == identity]
                ].mean()
                for identity in batch.sequence_ids
            ]
        )
        ece, reliability = _reliability_table(priors, errors, batch.scored_mask)
        counts = parameter_counts[name]
        outcomes = tuple(
            {
                "sequence_id": sequence_id,
                "round": round_index,
                "logical_failure": bool(failures[index]),
                "syndrome_valid": bool(recomputed_valid[index]),
                "converged": bool(decoded.converged[index]),
                "iterations": int(decoded.iterations[index]),
                "correction_weight": int(decoded.corrections[index].sum()),
                "predicted_observables": decoded.predicted_observables[index].astype(int).tolist(),
                "true_logical_flips": selected_logical[index].astype(int).tolist(),
                "observed_error_weight": int(selected_errors[index].sum()),
                "forecast_nll": float(round_nll[index]),
                "forecast_brier": float(round_brier[index]),
                "mean_predicted_probability": float(selected_priors[index].mean()),
                "setup_latency_seconds": float(decoded.setup_latency_seconds[index]),
                "decode_latency_seconds": float(decoded.decode_latency_seconds[index]),
            }
            for index, (sequence_id, round_index) in enumerate(membership)
        )
        arms[name] = ArmEvaluation(
            name=name,
            sequence_membership=membership,
            per_sequence_nll=per_sequence_nll,
            per_sequence_brier=per_sequence_brier,
            per_sequence_bler=per_sequence_bler,
            overall_nll=float(per_sequence_nll.mean()),
            overall_brier=float(per_sequence_brier.mean()),
            overall_bler=float(failures.mean()),
            convergence_rate=float(decoded.converged.mean()),
            mean_iterations=float(decoded.iterations.mean()),
            mean_correction_weight=float(decoded.corrections.sum(axis=1).mean()),
            latency_p50_seconds=float(np.quantile(decoded.latency_seconds, 0.50)),
            latency_p95_seconds=float(np.quantile(decoded.latency_seconds, 0.95)),
            latency_p99_seconds=float(np.quantile(decoded.latency_seconds, 0.99)),
            expected_calibration_error=float(ece),
            reliability=reliability,
            logical_failures=failures,
            syndrome_valid=recomputed_valid,
            converged=decoded.converged.copy(),
            iterations=decoded.iterations.copy(),
            correction_weights=decoded.corrections.sum(axis=1),
            setup_latency_seconds=decoded.setup_latency_seconds.copy(),
            decode_latency_seconds=decoded.decode_latency_seconds.copy(),
            stored_parameters=int(counts[0]),
            effective_parameters=int(counts[1]),
            per_round_outcomes=outcomes,
        )
    return PairedCausalEvaluation(batch.regime, batch.role, membership, arms)


def reduced_progression(
    *,
    regime: str,
    selected_name: str,
    stationary_nll: np.ndarray,
    selected_nll: np.ndarray,
    stationary_bler: np.ndarray,
    selected_bler: np.ndarray,
) -> ReducedProgression:
    """Apply the exact descriptive Gate-2 headroom rule for one B-E regime."""

    if regime == "stationary_iid":
        raise ValueError("Regime A is a negative control, not a B-E progression gate")
    values = [
        np.asarray(stationary_nll, dtype=np.float64),
        np.asarray(selected_nll, dtype=np.float64),
        np.asarray(stationary_bler, dtype=np.float64),
        np.asarray(selected_bler, dtype=np.float64),
    ]
    if any(value.ndim != 1 or value.size == 0 for value in values):
        raise ValueError("progression inputs must be nonempty per-sequence vectors")
    if any(value.shape != values[0].shape for value in values[1:]):
        raise ValueError("progression vectors must have identical sequence membership")
    nll_gain = float(values[0].mean() - values[1].mean())
    bler_difference = float(values[3].mean() - values[2].mean())
    return ReducedProgression(
        regime=regime,
        selected_name=selected_name,
        stationary_nll=tuple(map(float, values[0])),
        selected_nll=tuple(map(float, values[1])),
        stationary_bler=tuple(map(float, values[2])),
        selected_bler=tuple(map(float, values[3])),
        nll_improvement=nll_gain,
        bler_difference=bler_difference,
        progressed=nll_gain > 0.0 and bler_difference <= 0.0,
    )


def _interaction(losses: dict[str, np.ndarray]) -> np.ndarray:
    required = {"cnn_fir", "fno_fir", "cnn_hippo", "fno_hippo"}
    if set(losses) != required:
        raise ValueError(f"crossed losses must contain exactly {sorted(required)}")
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in losses.items()}
    reference = arrays["cnn_fir"]
    if (
        reference.ndim != 1
        or reference.size == 0
        or any(value.shape != reference.shape for value in arrays.values())
    ):
        raise ValueError("crossed losses must be aligned, nonempty per-sequence vectors")
    return (arrays["cnn_fir"] - arrays["fno_fir"]) - (arrays["cnn_hippo"] - arrays["fno_hippo"])


def reduced_factor_diagnostics(
    *,
    in_basis_losses: dict[str, np.ndarray],
    basis_mismatch_losses: dict[str, np.ndarray],
) -> ReducedFactorDiagnostics:
    """Report crossed interaction directions without a reduced-sample claim."""

    in_basis = _interaction(in_basis_losses)
    mismatch = _interaction(basis_mismatch_losses)
    in_basis_mean = float(in_basis.mean())
    mismatch_mean = float(mismatch.mean())
    return ReducedFactorDiagnostics(
        in_basis,
        mismatch,
        in_basis_mean,
        mismatch_mean,
        bool(in_basis_mean != 0.0 and np.sign(in_basis_mean) == np.sign(mismatch_mean)),
    )


def evaluate_oracle_sanity(
    probabilities: np.ndarray, targets: np.ndarray, scored_mask: np.ndarray
) -> dict[str, object]:
    """Score latent probabilities only as an unattainable generator sanity ceiling."""

    target = np.asarray(targets)
    mask = np.asarray(scored_mask)
    if mask.shape == (target.shape[1],):
        mask = np.broadcast_to(mask, target.shape[:2]).copy()
    nll = _forecast_nll_by_sequence(probabilities, target, mask)
    brier = _forecast_brier_by_sequence(probabilities, target, mask)
    return {
        "scope": "generator_sanity_only",
        "deployable_competition": False,
        "decoder_evaluated": False,
        "per_sequence_nll": tuple(map(float, nll)),
        "per_sequence_brier": tuple(map(float, brier)),
        "overall_nll": float(nll.mean()),
        "overall_brier": float(brier.mean()),
    }


__all__ = [
    "ArmEvaluation",
    "BaselineSelection",
    "CausalEvaluationBatch",
    "ForecastSplit",
    "PairedCausalEvaluation",
    "ReducedFactorDiagnostics",
    "ReducedProgression",
    "baseline_parameter_accounting",
    "evaluate_causal_arms",
    "evaluate_oracle_sanity",
    "fit_select_observation_baselines",
    "reduced_factor_diagnostics",
    "reduced_progression",
]
