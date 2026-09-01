"""Deterministic calibration of FNO correction probabilities."""

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

_PROBABILITY_EPSILON = 1e-5


@dataclass(frozen=True, slots=True)
class CalibrationParameters:
    """Scalar parameters for conditioning FNO logits on physical noise."""

    alpha: float
    beta: float
    temperature: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.alpha):
            raise ValueError("alpha must be finite")
        if not np.isfinite(self.beta):
            raise ValueError("beta must be finite")
        if not np.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")


@dataclass(frozen=True, slots=True)
class CalibrationScore:
    """Calibration-split outcome for one immutable parameter tuple."""

    parameters: CalibrationParameters
    invalid_count: int
    block_errors: int
    nll: float

    def __post_init__(self) -> None:
        for name, count in (
            ("invalid_count", self.invalid_count),
            ("block_errors", self.block_errors),
        ):
            if type(count) is not int or count < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not np.isfinite(self.nll):
            raise ValueError("nll must be finite")


CALIBRATION_GRID = tuple(
    CalibrationParameters(alpha=alpha, beta=beta, temperature=temperature)
    for alpha in (0.25, 0.5, 1.0, 2.0)
    for beta in (0.0, 0.5, 1.0)
    for temperature in (0.5, 1.0, 2.0, 4.0)
)


def deterministic_calibration_subset(
    rate_indices: Mapping[int, np.ndarray],
    *,
    max_shots: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Select a deterministic, near-balanced calibration-only decode subset."""
    if type(max_shots) is not int or max_shots <= 0:
        raise ValueError("max_shots must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not rate_indices:
        raise ValueError("at least one calibration rate is required")
    ranked: dict[int, list[int]] = {}
    for rate_index, raw_indices in sorted(rate_indices.items()):
        if type(rate_index) is not int or rate_index < 0:
            raise ValueError("calibration rate indices must be non-negative integers")
        indices = np.asarray(raw_indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0 or len(set(indices.tolist())) != indices.size:
            raise ValueError("calibration shot indices must be non-empty and unique per rate")
        ranked[rate_index] = sorted(
            indices.tolist(),
            key=lambda index: hashlib.sha256(
                f"{seed}:{rate_index}:{index}:calibration_decode".encode()
            ).digest(),
        )
    target = min(max_shots, sum(len(indices) for indices in ranked.values()))
    selected: dict[int, list[int]] = {rate_index: [] for rate_index in ranked}
    while sum(len(indices) for indices in selected.values()) < target:
        progressed = False
        for rate_index, indices in ranked.items():
            offset = len(selected[rate_index])
            if offset < len(indices):
                selected[rate_index].append(indices[offset])
                progressed = True
                if sum(len(values) for values in selected.values()) == target:
                    break
        if not progressed:
            raise RuntimeError("calibration subset allocation made no progress")
    subset = np.array(
        [index for rate_index in sorted(selected) for index in sorted(selected[rate_index])],
        dtype=np.int64,
    )
    digest = hashlib.sha256(subset.astype("<i8", copy=False).tobytes()).hexdigest()
    metadata: dict[str, object] = {
        "algorithm": "sha256_rank_within_rate_round_robin",
        "indices_sha256": digest,
        "max_shots": max_shots,
        "per_rate": [
            {"rate_index": rate_index, "shots": len(selected[rate_index])}
            for rate_index in sorted(selected)
        ],
        "seed": seed,
        "shots": int(subset.size),
    }
    return subset, metadata


def calibration_shortlists(
    screening_rows: Sequence[Mapping[str, object]],
    *,
    per_method: int,
) -> dict[str, list[int]]:
    """Build independent deterministic soft-prior and residual proxy shortlists."""
    if type(per_method) is not int or per_method <= 0:
        raise ValueError("per_method must be a positive integer")
    if not screening_rows or per_method > len(screening_rows):
        raise ValueError("per_method must not exceed the screening candidate count")

    def common(row: Mapping[str, object]) -> tuple[object, ...]:
        parameters = row.get("parameters")
        if not isinstance(parameters, Mapping):
            raise TypeError("calibration screening parameters are malformed")
        return (
            int(row["model_epoch"]),
            float(parameters["alpha"]),
            float(parameters["beta"]),
            float(parameters["temperature"]),
            int(row["work_index"]),
        )

    soft = sorted(
        screening_rows,
        key=lambda row: (float(row["nll"]), *common(row)),
    )[:per_method]
    residual = sorted(
        screening_rows,
        key=lambda row: (
            int(row["proposal_invalid_count"]),
            float(row["mean_residual_syndrome_weight"]),
            float(row["nll"]),
            *common(row),
        ),
    )[:per_method]
    return {
        "residual": [int(row["work_index"]) for row in residual],
        "soft_prior": [int(row["work_index"]) for row in soft],
    }


def _calibration_selection_key(
    row: Mapping[str, object], method: str
) -> tuple[object, ...]:
    score = row.get(method)
    parameters = row.get("parameters")
    if not isinstance(score, Mapping) or not isinstance(parameters, Mapping):
        raise TypeError("calibration candidate row is malformed")
    return (
        int(int(score["invalid_count"]) != 0),
        int(score["block_errors"]),
        float(score["nll"]),
        int(row["model_epoch"]),
        float(parameters["alpha"]),
        float(parameters["beta"]),
        float(parameters["temperature"]),
    )


def _selected_payload(row: Mapping[str, object], method: str) -> dict[str, object]:
    score = row.get(method)
    if not isinstance(score, Mapping):
        raise TypeError("calibration candidate score is malformed")
    return {
        **score,
        "inference_latency_seconds": row.get("inference_latency_seconds"),
        "model_checkpoint": row.get("model_checkpoint"),
        "model_epoch": row.get("model_epoch"),
        "parameters": row.get("parameters"),
        "screening_proxy": row.get("screening_proxy"),
        "work_index": row.get("work_index"),
    }


def validate_two_stage_calibration(
    screening_rows: object,
    hybrid_rows: object,
    shortlists: object,
    selected: object,
    *,
    checkpoint_candidates: object,
    candidates: Sequence[CalibrationParameters],
    screening_shots: int,
    syndrome_checks: int,
    decode_shots: int,
    shortlist_per_method: int,
) -> None:
    """Verify full deterministic screening, shortlist, decode, and argmin policy."""
    if not isinstance(checkpoint_candidates, list) or any(
        not isinstance(checkpoint, Mapping) for checkpoint in checkpoint_candidates
    ):
        raise TypeError("calibration checkpoint candidates are malformed")
    if not candidates or type(screening_shots) is not int or screening_shots <= 0:
        raise ValueError("calibration screening policy is malformed")
    if type(syndrome_checks) is not int or syndrome_checks <= 0:
        raise ValueError("calibration syndrome-check policy is malformed")
    if type(decode_shots) is not int or decode_shots <= 0:
        raise ValueError("calibration decode policy is malformed")
    if not isinstance(screening_rows, list) or len(screening_rows) != len(
        checkpoint_candidates
    ) * len(candidates):
        raise ValueError("calibration publication lacks the complete ordered screening grid")

    screening_fields = {
        "inference_latency_seconds",
        "logits_sha256",
        "mean_residual_syndrome_weight",
        "model_checkpoint",
        "model_epoch",
        "nll",
        "parameters",
        "proposal_invalid_count",
        "work_index",
    }
    checkpoint_measurements: dict[int, tuple[str, float]] = {}
    for work_index, row in enumerate(screening_rows):
        if not isinstance(row, Mapping) or set(row) != screening_fields:
            raise ValueError("calibration publication screening row is malformed")
        checkpoint_index = work_index // len(candidates)
        checkpoint = checkpoint_candidates[checkpoint_index]
        parameters = candidates[work_index % len(candidates)]
        latency = row.get("inference_latency_seconds")
        nll = row.get("nll")
        mean_residual = row.get("mean_residual_syndrome_weight")
        invalid = row.get("proposal_invalid_count")
        digest = row.get("logits_sha256")
        if (
            row.get("work_index") != work_index
            or row.get("model_epoch") != checkpoint.get("epoch")
            or row.get("model_checkpoint")
            != {"path": checkpoint.get("path"), "sha256": checkpoint.get("sha256")}
            or row.get("parameters") != asdict(parameters)
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or any(
                type(value) not in (int, float)
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in (latency, nll, mean_residual)
            )
            or float(mean_residual) > syndrome_checks
            or type(invalid) is not int
            or not 0 <= invalid <= screening_shots
        ):
            raise ValueError("calibration publication screening grid is inconsistent")
        measurement = (digest, float(latency))
        previous = checkpoint_measurements.setdefault(checkpoint_index, measurement)
        if previous != measurement:
            raise ValueError("calibration publication checkpoint screening diverges")

    expected_shortlists = calibration_shortlists(
        screening_rows,
        per_method=shortlist_per_method,
    )
    if shortlists != expected_shortlists:
        raise ValueError("calibration publication shortlists were not recomputed exactly")
    expected_union = sorted(
        set(expected_shortlists["soft_prior"]) | set(expected_shortlists["residual"])
    )
    if not isinstance(hybrid_rows, list) or [
        row.get("work_index") if isinstance(row, Mapping) else None for row in hybrid_rows
    ] != expected_union:
        raise ValueError("calibration publication does not decode the exact shortlist union")

    hybrid_fields = {
        "inference_latency_seconds",
        "logits_sha256",
        "model_checkpoint",
        "model_epoch",
        "parameters",
        "residual",
        "screening_proxy",
        "soft_prior",
        "work_index",
    }
    for row in hybrid_rows:
        if not isinstance(row, Mapping) or set(row) != hybrid_fields:
            raise ValueError("calibration publication hybrid row is malformed")
        work_index = int(row["work_index"])
        screening = screening_rows[work_index]
        latency = row.get("inference_latency_seconds")
        digest = row.get("logits_sha256")
        if (
            any(row.get(field) != screening.get(field) for field in (
                "model_checkpoint",
                "model_epoch",
                "parameters",
            ))
            or row.get("screening_proxy")
            != {
                "mean_residual_syndrome_weight": screening["mean_residual_syndrome_weight"],
                "nll": screening["nll"],
                "proposal_invalid_count": screening["proposal_invalid_count"],
            }
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(latency) not in (int, float)
            or not math.isfinite(float(latency))
            or float(latency) < 0.0
        ):
            raise ValueError("calibration publication hybrid provenance is inconsistent")
        method_nlls: list[float] = []
        for method in ("soft_prior", "residual"):
            score = row.get(method)
            if not isinstance(score, Mapping) or set(score) != {
                "block_errors",
                "invalid_count",
                "nll",
            }:
                raise ValueError(f"calibration publication {method} score is malformed")
            block_errors = score.get("block_errors")
            invalid_count = score.get("invalid_count")
            nll = score.get("nll")
            if (
                type(block_errors) is not int
                or type(invalid_count) is not int
                or not 0 <= invalid_count <= block_errors <= decode_shots
                or type(nll) not in (int, float)
                or not math.isfinite(float(nll))
                or float(nll) < 0.0
            ):
                raise ValueError(f"calibration publication {method} score is invalid")
            method_nlls.append(float(nll))
        if method_nlls[0] != method_nlls[1]:
            raise ValueError("calibration publication method NLL values diverge")

    if not isinstance(selected, Mapping) or set(selected) != {"soft_prior", "residual"}:
        raise ValueError("calibration publication selected methods are malformed")
    for method in ("soft_prior", "residual"):
        expected_row = min(
            (
                row
                for row in hybrid_rows
                if int(row["work_index"]) in expected_shortlists[method]
            ),
            key=lambda row: _calibration_selection_key(row, method),
        )
        if selected.get(method) != _selected_payload(expected_row, method):
            raise ValueError(f"calibration publication {method} is not the declared argmin")


def validate_calibration_progress_rows(
    rows: object,
    *,
    checkpoint_candidates: object,
    candidates: Sequence[CalibrationParameters],
    shots: int,
) -> list[dict[str, object]]:
    """Validate an exact ordered calibration-work prefix before resuming it."""
    if not isinstance(rows, list):
        raise TypeError("calibration progress candidates must be a list")
    if not isinstance(checkpoint_candidates, list) or any(
        not isinstance(checkpoint, dict) for checkpoint in checkpoint_candidates
    ):
        raise TypeError("calibration progress checkpoints are malformed")
    if type(shots) is not int or shots <= 0:
        raise ValueError("calibration progress shot count must be positive")
    if not candidates or len(rows) > len(checkpoint_candidates) * len(candidates):
        raise ValueError("calibration progress exceeds the declared work grid")

    checkpoint_measurements: dict[int, tuple[str, float]] = {}
    expected_row_fields = {
        "inference_latency_seconds",
        "logits_sha256",
        "model_checkpoint",
        "model_epoch",
        "parameters",
        "residual",
        "soft_prior",
    }
    for work_index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != expected_row_fields:
            raise ValueError("calibration progress candidate row is malformed")
        checkpoint_index = work_index // len(candidates)
        checkpoint = checkpoint_candidates[checkpoint_index]
        parameters = candidates[work_index % len(candidates)]
        if (
            row.get("model_epoch") != int(checkpoint["epoch"])
            or row.get("model_checkpoint")
            != {
                "path": str(checkpoint["path"]),
                "sha256": str(checkpoint["sha256"]),
            }
            or row.get("parameters") != asdict(parameters)
        ):
            raise ValueError("calibration progress candidate order is inconsistent")

        digest = row.get("logits_sha256")
        latency = row.get("inference_latency_seconds")
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(latency) not in (int, float)
            or not math.isfinite(latency)
            or latency < 0
        ):
            raise ValueError("calibration progress inference measurement is malformed")
        measurement = (digest, float(latency))
        previous = checkpoint_measurements.setdefault(checkpoint_index, measurement)
        if previous != measurement:
            raise ValueError("calibration progress checkpoint measurements diverge")

        score_nlls: list[float] = []
        for method in ("soft_prior", "residual"):
            score = row.get(method)
            if not isinstance(score, dict) or set(score) != {
                "block_errors",
                "invalid_count",
                "nll",
            }:
                raise ValueError(f"calibration progress {method} score is malformed")
            invalid_count = score["invalid_count"]
            block_errors = score["block_errors"]
            nll = score["nll"]
            if (
                type(invalid_count) is not int
                or type(block_errors) is not int
                or not 0 <= invalid_count <= block_errors <= shots
                or type(nll) not in (int, float)
                or not math.isfinite(nll)
                or nll < 0
            ):
                raise ValueError(f"calibration progress {method} score is invalid")
            score_nlls.append(float(nll))
        if score_nlls[0] != score_nlls[1]:
            raise ValueError("calibration progress method NLL values diverge")
    return rows


def calibrated_probabilities(
    logits: np.ndarray,
    error_rates: np.ndarray,
    parameters: CalibrationParameters,
) -> np.ndarray:
    """Apply a noise-conditioned sigmoid transform and clip decoder priors."""
    logits_array = np.asarray(logits, dtype=np.float64)
    rates = np.asarray(error_rates, dtype=np.float64)
    if logits_array.ndim < 1:
        raise ValueError("logits must include a shot dimension")
    if rates.ndim != 1 or rates.shape[0] != logits_array.shape[0]:
        raise ValueError("error_rates must have one value per logit shot")
    if not np.all(np.isfinite(rates)) or not np.all((0.0 < rates) & (rates < 1.0)):
        raise ValueError("error_rates must be finite probabilities between zero and one")
    logit_rates = np.log(rates) - np.log1p(-rates)
    rate_shape = (rates.shape[0],) + (1,) * (logits_array.ndim - 1)
    calibrated_logits = (
        parameters.alpha * logits_array / parameters.temperature
        + parameters.beta * logit_rates.reshape(rate_shape)
    )
    probabilities = _sigmoid(calibrated_logits)
    return np.clip(probabilities, _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON)


def select_calibration(scores: Sequence[CalibrationScore]) -> CalibrationScore:
    """Select the lexicographically best calibration-split candidate."""
    if not scores:
        raise ValueError("at least one calibration score is required")
    return min(
        scores,
        key=lambda score: (
            score.invalid_count,
            score.block_errors,
            score.nll,
            score.parameters.alpha,
            score.parameters.beta,
            score.parameters.temperature,
        ),
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Evaluate the sigmoid without overflow for extreme FNO logits."""
    probabilities = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_values = values[~positive]
    exponentials = np.exp(negative_values)
    probabilities[~positive] = exponentials / (1.0 + exponentials)
    return probabilities
