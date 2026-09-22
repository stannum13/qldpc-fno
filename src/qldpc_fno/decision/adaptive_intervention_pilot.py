"""Frozen, replayable adaptive intervention pilot with fully charged action work."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from qecsim import paulitools as pt
from qecsim.models.generic import DepolarizingErrorModel
from qecsim.models.planar import PlanarCode

from qldpc_fno.artifacts import sha256_file
from qldpc_fno.decision import adaptive_diagnostics as _diagnostics
from qldpc_fno.decision.planar_shot_accuracy import (
    _binary_vector,
    _stable_work,
    certify_reference,
    logical_class_recovery,
    score_recovery,
)
from qldpc_fno.decision.planar_shot_data import (
    _SOURCE_LABELS,
    _committed_digest,
    _file_binding,
    _git_provenance,
    _repository_root,
    _runtime_provenance,
    _shot_seed,
    _validate_producer,
    _validate_shot_manifest,
)
from qldpc_fno.decision.tensor_network import InvalidCosetMassError, planar_mps_coset_masses

_NOISE_MODEL = "qecsim_iid_depolarizing_code_capacity"
_SEED_DOMAIN = "qldpc-fno/adaptive-intervention-pilot/v3"
_FIXTURE_SEED_DOMAIN = "qldpc-fno/adaptive-intervention-pilot/test-fixture/v1"
_ERROR_RATES = (0.1, 0.15)
_REFERENCE_MODES = ("columns", "rows")

CAMPAIGN_SEED = 10044421296420932682
_FIXTURE_CAMPAIGN_SEED = 1727822112359709271
MARGIN_THRESHOLD = 0.30710401263493464
ACTION_IDS = (
    "columns_chi2",
    "rows_chi2",
    "columns_chi4",
    "rows_chi4",
    "columns_chi8",
    "rows_chi8",
    "columns_chi16",
    "rows_chi16",
    "columns_tol03",
    "rows_tol03",
    "columns_tol01",
    "rows_tol01",
    "columns_tol003",
    "rows_tol003",
)
PROBE_ACTION_IDS = ("columns_tol01", "rows_tol01")
_ACTION_TUPLES = (
    ("columns", 2, None),
    ("rows", 2, None),
    ("columns", 4, None),
    ("rows", 4, None),
    ("columns", 8, None),
    ("rows", 8, None),
    ("columns", 16, None),
    ("rows", 16, None),
    ("columns", None, 0.03),
    ("rows", None, 0.03),
    ("columns", None, 0.01),
    ("rows", None, 0.01),
    ("columns", None, 0.003),
    ("rows", None, 0.003),
)
_PILOT_CONFIG_KEYS = {
    "schema_version",
    "pilot_id",
    "required_shot_seed_domain",
    "required_shots_per_rate",
    "required_error_rates",
    "code_distance",
    "noise_model",
    "reference_modes",
    "reference_probability_tolerance",
    "reference_log_ratio_tolerance",
    "margin_threshold",
    "positive_gain_threshold",
    "oracle_tie_relative_tolerance",
    "oracle_tie_absolute_tolerance",
    "efficiency_unique_relative_separation",
    "probe_action_ids",
    "actions",
}
_SHOT_CONFIG_KEYS = {
    "schema_version",
    "seed_domain",
    "campaign_seed",
    "code_distance",
    "error_rates",
    "shots_per_rate",
    "noise_model",
}
_ACTION_KEYS = {"action_id", "mode", "chi", "tol"}
_SHOT_PROVENANCE_KEYS = {
    "qecsim_version",
    "git_commit",
    "git_dirty",
    "config_sha256",
    "config_path",
    "config_scope",
    "config_committed",
    "config_content",
    "source_sha256",
    "dependencies",
    "matching_backend",
}


@dataclass(frozen=True)
class ActionConfig:
    """One independently executed approximate tensor-contraction action."""

    action_id: str
    mode: str
    chi: int | None
    tol: float | None


@dataclass(frozen=True)
class PilotConfig:
    """Strictly parsed, immutable adaptive-intervention pilot policy."""

    schema_version: int
    pilot_id: str
    required_shot_seed_domain: str
    required_shots_per_rate: int
    required_error_rates: tuple[float, float]
    code_distance: int
    noise_model: str
    reference_modes: tuple[str, str]
    reference_probability_tolerance: float
    reference_log_ratio_tolerance: float
    margin_threshold: float
    positive_gain_threshold: float
    oracle_tie_relative_tolerance: float
    oracle_tie_absolute_tolerance: float
    efficiency_unique_relative_separation: float
    probe_action_ids: tuple[str, str]
    actions: tuple[ActionConfig, ...]


@dataclass(frozen=True)
class ShotConfig:
    """Strictly parsed immutable physical-shot generator configuration."""

    schema_version: int
    seed_domain: str
    campaign_seed: int
    code_distance: int
    error_rates: tuple[float, float]
    shots_per_rate: int
    noise_model: str


@dataclass(frozen=True)
class CandidateOpportunity:
    """A valid-or-invalid candidate's posterior improvement and full cost."""

    action_id: str
    valid: bool
    gain: float | None
    composite_work: int

    def __post_init__(self) -> None:
        if self.action_id not in ACTION_IDS or self.action_id in PROBE_ACTION_IDS:
            raise ValueError("candidate action_id must be a frozen action")
        if type(self.valid) is not bool:
            raise TypeError("candidate validity must be boolean")
        if self.gain is not None and (
            isinstance(self.gain, bool) or not math.isfinite(float(self.gain))
        ):
            raise ValueError("candidate gain must be finite or null")
        if type(self.composite_work) is not int or self.composite_work < 0:
            raise ValueError("candidate composite work must be a nonnegative integer")


@dataclass(frozen=True)
class OracleSelection:
    """One deterministic offline oracle decision, or its explicit null result."""

    action_id: str | None
    gain: float | None
    composite_work: int | None
    efficiency: float | None
    uniquely_separated: bool


def normalized_probabilities(probabilities: Sequence[float]) -> tuple[float, float, float, float]:
    """Return an immutable normalized four-logical-class posterior."""
    normalized = _diagnostics._probabilities(probabilities)
    return tuple(float(value) for value in normalized)  # type: ignore[return-value]


def probability_margin(probabilities: Sequence[float]) -> float:
    """Return the stable normalized winning-class probability margin."""
    return _diagnostics.probability_margin(probabilities)


def total_variation(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the stable total-variation distance between logical posteriors."""
    return _diagnostics.total_variation(left, right)


def jensen_shannon_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the stable natural-log Jensen-Shannon divergence in nats."""
    return _diagnostics.jensen_shannon_divergence(left, right)


def reference_conditional_excess_risk(
    candidate: Sequence[float], reference: Sequence[float]
) -> float:
    """Return the extra reference-conditional MAP decision risk of ``candidate``.

    The certified reference supplies both the posterior and its Bayes-optimal
    class. The candidate only supplies the class it would select.
    """
    candidate_posterior = _diagnostics._probabilities(candidate)
    reference_posterior = _diagnostics._probabilities(reference)
    candidate_class = int(candidate_posterior.argmax())
    return float(reference_posterior.max() - reference_posterior[candidate_class])


def symmetric_reference_posterior(
    columns: Sequence[float], rows: Sequence[float]
) -> tuple[float, float, float, float]:
    """Average separately normalized unrestricted views, then normalize again."""
    mean = 0.5 * (_diagnostics._probabilities(columns) + _diagnostics._probabilities(rows))
    return normalized_probabilities(mean)


def _valid_selected_class(selected_class: int | None) -> bool:
    return type(selected_class) is int and 0 <= selected_class < 4


def margin_gate_accepts(
    columns: Sequence[float] | None,
    columns_selected_class: int | None,
    rows: Sequence[float] | None,
    rows_selected_class: int | None,
    *,
    threshold: float = MARGIN_THRESHOLD,
) -> bool:
    """Accept only agreeing valid probes whose smaller margin strictly clears the gate."""
    if columns is None or rows is None:
        return False
    if not _valid_selected_class(columns_selected_class) or not _valid_selected_class(
        rows_selected_class
    ):
        return False
    if columns_selected_class != rows_selected_class:
        return False
    if not math.isfinite(threshold):
        raise ValueError("margin threshold must be finite")
    minimum_margin = min(probability_margin(columns), probability_margin(rows))
    return minimum_margin > threshold


def _work_value(work: int, name: str) -> int:
    if type(work) is not int or work < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return work


def composite_work(columns_probe_work: int, rows_probe_work: int, candidate_work: int) -> int:
    """Charge both independent probes and the independently run candidate."""
    return (
        _work_value(columns_probe_work, "columns_probe_work")
        + _work_value(rows_probe_work, "rows_probe_work")
        + _work_value(candidate_work, "candidate_work")
    )


def candidate_benefit(
    columns_probe: Sequence[float] | None,
    rows_probe: Sequence[float] | None,
    candidate: Sequence[float] | None,
    reference: Sequence[float],
) -> float | None:
    """Return candidate TV reduction from the best available probe, or null.

    A lone valid probe supplies the baseline. With no valid probe or candidate
    there is no posterior-target utility, even though all attempted work remains
    chargeable elsewhere.
    """
    if candidate is None:
        return None
    probe_distances = [
        total_variation(probe, reference)
        for probe in (columns_probe, rows_probe)
        if probe is not None
    ]
    if not probe_distances:
        return None
    return float(min(probe_distances) - total_variation(candidate, reference))


def _eligible_positive_candidates(
    candidates: Sequence[CandidateOpportunity], *, require_work: bool
) -> list[CandidateOpportunity]:
    return [
        candidate
        for candidate in candidates
        if candidate.valid
        and candidate.gain is not None
        and candidate.gain > 1e-6
        and (not require_work or candidate.composite_work > 0)
    ]


def _literal_action_rank(action_id: str) -> int:
    return ACTION_IDS.index(action_id)


def _select_by_metric(
    candidates: Sequence[CandidateOpportunity], metric: Callable[[CandidateOpportunity], float]
) -> CandidateOpportunity:
    """Choose globally tied maximum contenders by work then frozen action order."""
    maximum_metric = max(metric(candidate) for candidate in candidates)
    contenders = [
        candidate
        for candidate in candidates
        if math.isclose(
            metric(candidate),
            maximum_metric,
            rel_tol=1e-12,
            abs_tol=0.0,
        )
    ]
    return min(
        contenders,
        key=lambda candidate: (candidate.composite_work, _literal_action_rank(candidate.action_id)),
    )


def _selection(candidates: Sequence[CandidateOpportunity], *, efficiency: bool) -> OracleSelection:
    eligible = _eligible_positive_candidates(candidates, require_work=efficiency)
    if not eligible:
        return OracleSelection(None, None, None, None, False)
    if efficiency:

        def metric(candidate: CandidateOpportunity) -> float:
            return float(candidate.gain) / candidate.composite_work
    else:

        def metric(candidate: CandidateOpportunity) -> float:
            return float(candidate.gain)

    winner = _select_by_metric(eligible, metric)
    winner_metric = float(metric(winner))
    comparators = (
        eligible
        if efficiency
        else [
            candidate for candidate in candidates if candidate.valid and candidate.gain is not None
        ]
    )
    runner_metrics = [
        float(metric(candidate))
        for candidate in comparators
        if candidate.action_id != winner.action_id
    ]
    if not runner_metrics:
        uniquely_separated = True
    elif efficiency:
        runner_metric = max(runner_metrics)
        uniquely_separated = (winner_metric - runner_metric) / runner_metric > 0.01
    else:
        uniquely_separated = winner_metric - max(runner_metrics) > 1e-6
    return OracleSelection(
        action_id=winner.action_id,
        gain=float(winner.gain),
        composite_work=winner.composite_work,
        efficiency=winner_metric if efficiency else None,
        uniquely_separated=uniquely_separated,
    )


def select_accuracy_oracle(candidates: Sequence[CandidateOpportunity]) -> OracleSelection:
    """Choose the positive-gain candidate with the greatest TV reduction."""
    return _selection(candidates, efficiency=False)


def select_efficiency_oracle(candidates: Sequence[CandidateOpportunity]) -> OracleSelection:
    """Choose the positive-gain candidate with the greatest gain per full FLOP cost."""
    return _selection(candidates, efficiency=True)


def class_mismatch_transition(
    baseline_class: int, candidate_class: int, reference_class: int
) -> str:
    """Label whether a candidate repairs or introduces a reference-class mismatch."""
    if not all(
        _valid_selected_class(value) for value in (baseline_class, candidate_class, reference_class)
    ):
        raise ValueError("logical classes must be integers from 0 through 3")
    baseline_mismatch = baseline_class != reference_class
    candidate_mismatch = candidate_class != reference_class
    if baseline_mismatch and not candidate_mismatch:
        return "repaired_mismatch"
    if not baseline_mismatch and candidate_mismatch:
        return "introduced_mismatch"
    if baseline_mismatch:
        return "persistent_mismatch"
    return "matched"


def physical_outcome_transition(baseline_discordance: bool, candidate_discordance: bool) -> str:
    """Label repair, introduction, or persistence of physical-outcome discordance."""
    if type(baseline_discordance) is not bool or type(candidate_discordance) is not bool:
        raise TypeError("outcome discordance values must be boolean")
    if baseline_discordance and not candidate_discordance:
        return "repaired_discordance"
    if not baseline_discordance and candidate_discordance:
        return "introduced_discordance"
    if baseline_discordance:
        return "persistent_discordance"
    return "agreed"


def compact_spectral_summary(work: Mapping | None, *, available: bool) -> dict[str, object]:
    """Summarize only the completed action's recorded spectra, never future runs."""
    events = work.get("truncation_events", []) if work is not None and available else []
    spectra = [spectrum for event in events for spectrum in event["spectral_summaries"]]
    return {
        "available": available,
        "truncation_event_count": len(events),
        "spectrum_count": len(spectra),
        "maximum_discarded_squared_weight_fraction": max(
            (s["discarded_squared_weight_fraction"] for s in spectra), default=None
        ),
        "mean_spectral_entropy": (
            sum(s["spectral_entropy"] for s in spectra) / len(spectra) if spectra else None
        ),
        "maximum_retained_rank": max((s["retained_rank"] for s in spectra), default=None),
        "estimated_arithmetic_flops": (
            int(work["estimated_arithmetic_flops"]) if available and work is not None else None
        ),
    }


def _run_action(
    action: ActionConfig,
    shot: Mapping,
    config: PilotConfig,
    *,
    timing_records: list[dict] | None = None,
) -> dict:
    started = perf_counter() if timing_records is not None else None
    try:
        result = planar_mps_coset_masses(
            rows=config.code_distance,
            columns=config.code_distance,
            syndrome=shot["syndrome"],
            error_rate=shot["error_rate"],
            chi=action.chi,
            tol=action.tol,
            mode=action.mode,
            trace_work=True,
        )
    except InvalidCosetMassError as error:
        work = _stable_work(error.work)
        record = {
            "valid": False,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "probabilities": None,
            "selected_class": None,
            "work": work,
        }
    else:
        work = _stable_work(result.work)
        record = {
            "valid": True,
            "exception_type": None,
            "exception_message": None,
            "masses": result.masses.tolist(),
            "probabilities": list(normalized_probabilities(result.probabilities)),
            "selected_class": result.selected_class,
            "work": work,
        }
    record.update(asdict(action))
    record["estimated_arithmetic_flops"] = int((work or {}).get("estimated_arithmetic_flops", 0))
    record["spectral_summary"] = compact_spectral_summary(work, available=record["valid"])
    if timing_records is not None:
        elapsed = perf_counter() - started
        timing_records.append(
            {
                "action_id": action.action_id,
                "valid": record["valid"],
                "wall_seconds": elapsed if record["valid"] else None,
                "exception_type": record["exception_type"],
                "unavailable_reason": None if record["valid"] else "invalid_contraction",
            }
        )
    return record


def _physical_label(code: PlanarCode, shot: Mapping, selected_class: int | None) -> dict:
    if selected_class is None:
        return {
            "recovery_bsf": None,
            "syndrome_valid": False,
            "logical_failure": True,
            "logical_signature": None,
            "recovery_syndrome": None,
            "residual_syndrome": None,
            "residual": None,
            "failure": True,
        }
    recovery = logical_class_recovery(code, shot["syndrome"], selected_class)
    score = score_recovery(code, shot["error"], shot["syndrome"], recovery)
    return {
        **{
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in score.items()
        },
        "recovery_bsf": recovery.tolist(),
        "failure": not score["syndrome_valid"] or score["logical_failure"],
    }


def _pre_action_features(shot: Mapping, probes: Sequence[dict]) -> dict:
    columns, rows = probes
    valid = all(probe["valid"] for probe in probes)
    margins = [probability_margin(p["probabilities"]) if p["valid"] else None for p in probes]
    return {
        "error_rate": float(shot["error_rate"]),
        "syndrome_hamming_weight": int(np.count_nonzero(shot["syndrome"])),
        "probes": {
            p["action_id"]: {
                "valid": p["valid"],
                "selected_class": p["selected_class"],
                "margin": margin,
                "work": (
                    {
                        key: value
                        for key, value in p["work"].items()
                        if not isinstance(value, (list, dict))
                    }
                    if p["valid"]
                    else None
                ),
                "spectral_summary": p["spectral_summary"],
            }
            for p, margin in zip(probes, margins, strict=True)
        },
        "class_agreement": columns["selected_class"] == rows["selected_class"] if valid else None,
        "minimum_margin": min(margins) if valid else None,
        "cross_probe_total_variation": total_variation(
            columns["probabilities"], rows["probabilities"]
        )
        if valid
        else None,
        "cross_probe_jensen_shannon": jensen_shannon_divergence(
            columns["probabilities"], rows["probabilities"]
        )
        if valid
        else None,
    }


def evaluate_shot(
    shot: Mapping[str, object],
    config: PilotConfig,
    *,
    timing_records: list[dict] | None = None,
) -> dict[str, object]:
    """Evaluate one joined physical shot, retaining every independent run's work.

    ``features`` contains only frozen probe observations. Physical outcomes,
    reference targets, retrospective utilities and oracles live under ``labels``.
    Caller-supplied shot identity is copied alongside error and syndrome; this
    function neither samples errors nor writes artifacts. Optional timing is
    appended out of band and never enters the returned inference or oracle data.
    """
    code = PlanarCode(config.code_distance, config.code_distance)
    syndrome = _binary_vector(shot["syndrome"], length=len(code.stabilizers), name="syndrome")
    error = _binary_vector(shot["error"], length=2 * code.n_k_d[0], name="error")
    if not np.array_equal(pt.bsp(error, code.stabilizers.T), syndrome):
        raise ValueError("physical error and syndrome are not joined")
    if shot["error_rate"] not in config.required_error_rates:
        raise ValueError("shot error_rate is not a frozen pilot rate")
    references = {
        mode: _run_action(
            ActionConfig(f"exact_{mode}", mode, None, None),
            shot,
            config,
            timing_records=timing_records,
        )
        for mode in config.reference_modes
    }
    certificate, reference_error, reference_message = None, None, None
    if all(view["valid"] for view in references.values()):
        try:
            certificate = certify_reference(
                np.asarray(references["columns"]["masses"]),
                np.asarray(references["rows"]["masses"]),
            )
        except ValueError as invalid:
            # certify_reference's declared failure contract is numerical ValueError.
            reference_error, reference_message = type(invalid).__name__, str(invalid)
    else:
        reference_error = "invalid_reference_view"
    target = (
        symmetric_reference_posterior(
            references["columns"]["probabilities"], references["rows"]["probabilities"]
        )
        if certificate is not None
        else None
    )
    reference_class = certificate["selected_class"] if certificate is not None else None
    reference_physical = (
        _physical_label(code, shot, reference_class) if target is not None else None
    )
    actions = {
        action.action_id: _run_action(action, shot, config, timing_records=timing_records)
        for action in config.actions
    }
    probes = [actions[action_id] for action_id in config.probe_action_ids]
    columns, rows = probes
    accepted = margin_gate_accepts(
        columns["probabilities"],
        columns["selected_class"],
        rows["probabilities"],
        rows["selected_class"],
        threshold=config.margin_threshold,
    )
    action_labels = {}
    for action_id, action in actions.items():
        physical = _physical_label(code, shot, action["selected_class"])
        comparable = target is not None and action["valid"]
        action_labels[action_id] = {
            "physical": physical,
            "class_mismatch": action["selected_class"] != reference_class if comparable else None,
            "physical_outcome_discordance": (
                physical["failure"] != reference_physical["failure"] if comparable else None
            ),
            "total_variation": total_variation(action["probabilities"], target)
            if comparable
            else None,
            "jensen_shannon_divergence": jensen_shannon_divergence(action["probabilities"], target)
            if comparable
            else None,
            "reference_conditional_excess_risk": reference_conditional_excess_risk(
                action["probabilities"], target
            )
            if comparable
            else None,
        }
    valid_probes = [p for p in probes if p["valid"]]
    baseline = (
        min(valid_probes, key=lambda p: total_variation(p["probabilities"], target))
        if (target is not None and valid_probes)
        else None
    )
    baseline_tv = (
        total_variation(baseline["probabilities"], target) if baseline is not None else None
    )
    candidates, opportunities = {}, []
    for action_id, action in actions.items():
        if action_id in config.probe_action_ids:
            continue
        gain = (
            candidate_benefit(
                columns["probabilities"], rows["probabilities"], action["probabilities"], target
            )
            if target is not None
            else None
        )
        work = composite_work(
            *(p["estimated_arithmetic_flops"] for p in probes), action["estimated_arithmetic_flops"]
        )
        opportunities.append(CandidateOpportunity(action_id, action["valid"], gain, work))
        comparable = baseline is not None and action["valid"]
        candidates[action_id] = {
            "gain": gain,
            "composite_work": work,
            "class_mismatch_transition": class_mismatch_transition(
                baseline["selected_class"], action["selected_class"], reference_class
            )
            if comparable
            else None,
            "physical_outcome_transition": physical_outcome_transition(
                action_labels[baseline["action_id"]]["physical_outcome_discordance"],
                action_labels[action_id]["physical_outcome_discordance"],
            )
            if comparable
            else None,
            "physical_outcome_changed": (
                action_labels[baseline["action_id"]]["physical"]["failure"]
                != action_labels[action_id]["physical"]["failure"]
            )
            if comparable
            else None,
        }
    return {
        "shot": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in shot.items()
        },
        "features": _pre_action_features(shot, probes),
        "gate": {"accepted": accepted, "escalated": not accepted},
        "actions": actions,
        "reference": {
            "views": references,
            "certified": certificate is not None,
            "certificate": certificate,
            "exception_type": reference_error,
            "exception_message": reference_message,
        },
        "labels": {
            "reference_probabilities": list(target) if target is not None else None,
            "reference_physical": reference_physical,
            "actions": action_labels,
            "baseline_action_id": baseline["action_id"] if baseline is not None else None,
            "baseline_total_variation": baseline_tv,
            "candidates": candidates,
            "accuracy_oracle": asdict(
                select_accuracy_oracle(opportunities if not accepted else [])
            ),
            "efficiency_oracle": asdict(
                select_efficiency_oracle(opportunities if not accepted else [])
            ),
        },
        "total_estimated_arithmetic_flops": sum(
            action["estimated_arithmetic_flops"]
            for action in (*references.values(), *actions.values())
        ),
        "invalid_estimated_arithmetic_flops": sum(
            action["estimated_arithmetic_flops"]
            for action in (*references.values(), *actions.values())
            if not action["valid"]
        ),
    }


def _mean(values: Sequence) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _reference_work_summary(views: Sequence[dict]) -> dict:
    """Account for each attempted view, independently of two-view certification."""
    valid = sum(view["valid"] for view in views)
    work = [view["estimated_arithmetic_flops"] for view in views]
    return {
        "attempted_shots": len(views),
        "valid_shots": valid,
        "invalid_shots": len(views) - valid,
        "validity_rate": _mean([view["valid"] for view in views]),
        "estimated_arithmetic_flops_values": work,
        "estimated_arithmetic_flops_total": sum(work),
        "estimated_arithmetic_flops_mean": _mean(work),
        "invalid_estimated_arithmetic_flops_total": sum(
            view["estimated_arithmetic_flops"] for view in views if not view["valid"]
        ),
    }


def _rate_summary(results: Sequence[dict], rate: float) -> dict:
    certified = sum(r["reference"]["certified"] for r in results)
    summary = {
        "error_rate": rate,
        "attempted_shots": len(results),
        "certified_reference_shots": certified,
        "invalid_reference_shots": len(results) - certified,
        "gate_acceptance_rate": _mean([r["gate"]["accepted"] for r in results]),
        "escalated_shots": sum(r["gate"]["escalated"] for r in results),
        "actions": {},
        "candidates": {},
        "references": {
            f"exact_{mode}": _reference_work_summary(
                [r["reference"]["views"][mode] for r in results]
            )
            for mode in _REFERENCE_MODES
        },
        "total_estimated_arithmetic_flops": sum(
            r["total_estimated_arithmetic_flops"] for r in results
        ),
        "invalid_estimated_arithmetic_flops": sum(
            r["invalid_estimated_arithmetic_flops"] for r in results
        ),
    }
    for action_id in ACTION_IDS:
        labels = [r["labels"]["actions"][action_id] for r in results]
        action = {
            "attempted_shots": len(results),
            "certified_reference_shots": certified,
            "valid_shots": sum(r["actions"][action_id]["valid"] for r in results),
            "validity_rate": _mean([r["actions"][action_id]["valid"] for r in results]),
            "physical_failure_rate": _mean([label["physical"]["failure"] for label in labels]),
            "estimated_arithmetic_flops_mean": _mean(
                [r["actions"][action_id]["estimated_arithmetic_flops"] for r in results]
            ),
            "estimated_arithmetic_flops_values": [
                r["actions"][action_id]["estimated_arithmetic_flops"] for r in results
            ],
        }
        for metric in (
            "total_variation",
            "jensen_shannon_divergence",
            "class_mismatch",
            "physical_outcome_discordance",
            "reference_conditional_excess_risk",
        ):
            distribution = [label[metric] for label in labels if label[metric] is not None]
            action[metric + "_values"] = distribution
            action[metric + "_observed_shots"] = len(distribution)
            action[metric + "_mean"] = _mean(distribution)
        for oracle in ("accuracy_oracle", "efficiency_oracle"):
            action[oracle + "_wins"] = sum(
                r["labels"][oracle]["action_id"] == action_id for r in results
            )
            action[oracle + "_win_rate"] = _mean(
                [r["labels"][oracle]["action_id"] == action_id for r in results]
            )
        summary["actions"][action_id] = action
        if action_id not in PROBE_ACTION_IDS:
            candidates = [
                r["labels"]["candidates"][action_id] for r in results if r["gate"]["escalated"]
            ]
            gains = [candidate["gain"] for candidate in candidates if candidate["gain"] is not None]
            costs = [candidate["composite_work"] for candidate in candidates]
            summary["candidates"][action_id] = {
                "escalated_shots": len(candidates),
                "observed_shots": len(gains),
                "gain_values": gains,
                "gain_mean": _mean(gains),
                "composite_work_values": costs,
                "composite_work_mean": _mean(costs),
            }
    return summary


def summarize_evaluations(results: Sequence[dict], config: PilotConfig) -> dict[str, object]:
    """Describe paired shot results by rate and equal-rate means, without inference.

    Missing posterior observations are explicit. An absent rate-level estimate
    produces a null equal-rate estimate, never a complete-case pooled substitute.
    Advancement is permitted only for the full frozen pilot with all references.
    """
    per_rate = [
        _rate_summary([r for r in results if r["shot"]["error_rate"] == rate], rate)
        for rate in config.required_error_rates
    ]
    if sum(rate["attempted_shots"] for rate in per_rate) != len(results):
        raise ValueError("unexpected physical error rate in evaluated shots")
    identities = [(r["shot"]["error_rate"], r["shot"]["shot_id"]) for r in results]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate physical shot in summary")
    certified = sum(rate["certified_reference_shots"] for rate in per_rate)
    complete = all(rate["attempted_shots"] == config.required_shots_per_rate for rate in per_rate)
    all_certified = certified == len(results) and complete
    unique_winners = sorted(
        {
            r["labels"]["efficiency_oracle"]["action_id"]
            for r in results
            if r["gate"]["escalated"]
            and r["labels"]["efficiency_oracle"]["uniquely_separated"]
            and r["labels"]["efficiency_oracle"]["action_id"] is not None
        },
        key=ACTION_IDS.index,
    )
    equal_rate = {"actions": {}, "candidates": {}, "references": {}}
    gate_rates = [rate["gate_acceptance_rate"] for rate in per_rate]
    equal_rate["gate_acceptance_rate"] = _mean(gate_rates) if None not in gate_rates else None
    for mode in _REFERENCE_MODES:
        # Counts and totals describe actual attempts. Means and empirical weights
        # describe an equal-rate mixture, even for unequal diagnostic sample sizes.
        reference = _reference_work_summary(
            [
                r["reference"]["views"][mode]
                for rate in config.required_error_rates
                for r in results
                if r["shot"]["error_rate"] == rate
            ]
        )
        for metric in ("estimated_arithmetic_flops_mean", "validity_rate"):
            values = [rate["references"][f"exact_{mode}"][metric] for rate in per_rate]
            reference[metric] = _mean(values) if None not in values else None
        counts = [rate["attempted_shots"] for rate in per_rate]
        reference["estimated_arithmetic_flops_weights"] = (
            [1 / (len(counts) * count) for count in counts for _ in range(count)]
            if all(counts)
            else None
        )
        equal_rate["references"][f"exact_{mode}"] = reference
    for action_id in ACTION_IDS:
        action = {}
        for key in per_rate[0]["actions"][action_id]:
            if key.endswith(("_mean", "_rate")):
                values = [rate["actions"][action_id][key] for rate in per_rate]
                action[key] = _mean(values) if None not in values else None
        equal_rate["actions"][action_id] = action
        if action_id not in PROBE_ACTION_IDS:
            candidate = {}
            for key in ("gain_mean", "composite_work_mean"):
                values = [rate["candidates"][action_id][key] for rate in per_rate]
                candidate[key] = _mean(values) if None not in values else None
            equal_rate["candidates"][action_id] = candidate
    return {
        "statistical_unit": "physical_shot",
        "attempted_shots": len(results),
        "certified_reference_shots": certified,
        "invalid_reference_shots": len(results) - certified,
        "per_rate": per_rate,
        "equal_rate": equal_rate,
        "advancement": {
            "complete_pilot": complete,
            "margin_gate_clause": all_certified and all(value >= 0.5 for value in gate_rates),
            "heterogeneous_efficiency_clause": all_certified and len(unique_winners) >= 2,
            "uniquely_best_efficiency_action_ids": unique_winners,
        },
    }


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {value}")


def _load_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            Path(path).read_text(),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON configuration: {path}") from error
    if not isinstance(value, dict):
        raise TypeError("configuration must be a JSON object")
    return value


def _finite_float(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _positive_finite_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _exact_fields(payload: dict[str, object], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{name} must contain exactly its frozen schema fields")


def _parse_actions(value: object) -> tuple[ActionConfig, ...]:
    if not isinstance(value, list) or len(value) != len(ACTION_IDS):
        raise ValueError("actions must contain exactly 14 frozen actions")
    actions: list[ActionConfig] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise TypeError("each action must be an object")
        _exact_fields(item, _ACTION_KEYS, "action")
        action_id, mode, chi, tolerance = (
            item["action_id"],
            item["mode"],
            item["chi"],
            item["tol"],
        )
        if not isinstance(action_id, str) or mode not in _REFERENCE_MODES:
            raise ValueError("action_id and mode must be valid")
        if chi is not None and (type(chi) is not int or chi <= 0):
            raise ValueError("action chi must be a positive integer or null")
        tol = None if tolerance is None else _positive_finite_float(tolerance, "action tol")
        action = ActionConfig(action_id=action_id, mode=mode, chi=chi, tol=tol)
        if (action.action_id, (action.mode, action.chi, action.tol)) != (
            ACTION_IDS[index],
            _ACTION_TUPLES[index],
        ):
            raise ValueError("actions must use the frozen literal order and parameters")
        actions.append(action)
    if len({(action.mode, action.chi, action.tol) for action in actions}) != len(actions):
        raise ValueError("action tuples must be unique")
    return tuple(actions)


def load_pilot_config(path: Path) -> PilotConfig:
    """Load only the preregistered adaptive-intervention pilot configuration."""
    payload = _load_object(path)
    _exact_fields(payload, _PILOT_CONFIG_KEYS, "pilot configuration")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["pilot_id"] != "adaptive_intervention_pilot_v3"
    ):
        raise ValueError("unsupported pilot schema or identity")
    if payload["required_shot_seed_domain"] != _SEED_DOMAIN:
        raise ValueError("required_shot_seed_domain must use the frozen pilot domain")
    if (
        type(payload["required_shots_per_rate"]) is not int
        or payload["required_shots_per_rate"] != 64
    ):
        raise ValueError("required_shots_per_rate must be exactly 64")
    rates = payload["required_error_rates"]
    if (
        not isinstance(rates, list)
        or tuple(_finite_float(rate, "required_error_rates") for rate in rates) != _ERROR_RATES
    ):
        raise ValueError("required_error_rates must be exactly [0.1, 0.15]")
    if (
        type(payload["code_distance"]) is not int
        or payload["code_distance"] != 5
        or payload["noise_model"] != _NOISE_MODEL
    ):
        raise ValueError("pilot code identity must be the frozen planar noise model")
    modes = payload["reference_modes"]
    if not isinstance(modes, list) or tuple(modes) != _REFERENCE_MODES:
        raise ValueError("reference_modes must be columns then rows")
    probability_tolerance = _positive_finite_float(
        payload["reference_probability_tolerance"], "reference_probability_tolerance"
    )
    log_ratio_tolerance = _positive_finite_float(
        payload["reference_log_ratio_tolerance"], "reference_log_ratio_tolerance"
    )
    threshold = _finite_float(payload["margin_threshold"], "margin_threshold")
    positive_gain = _positive_finite_float(
        payload["positive_gain_threshold"], "positive_gain_threshold"
    )
    tie_tolerance = _positive_finite_float(
        payload["oracle_tie_relative_tolerance"], "oracle_tie_relative_tolerance"
    )
    absolute_tie_tolerance = _finite_float(
        payload["oracle_tie_absolute_tolerance"], "oracle_tie_absolute_tolerance"
    )
    efficiency_separation = _positive_finite_float(
        payload["efficiency_unique_relative_separation"],
        "efficiency_unique_relative_separation",
    )
    if (
        probability_tolerance != 1e-10
        or log_ratio_tolerance != 1e-8
        or threshold != MARGIN_THRESHOLD
        or positive_gain != 1e-6
        or tie_tolerance != 1e-12
        or absolute_tie_tolerance != 0.0
        or efficiency_separation != 0.01
    ):
        raise ValueError("pilot numerical constants must match the frozen values")
    probes = payload["probe_action_ids"]
    if probes != ["columns_tol01", "rows_tol01"]:
        raise ValueError("probe_action_ids must contain the two tol=0.01 views in order")
    actions = _parse_actions(payload["actions"])
    return PilotConfig(
        schema_version=1,
        pilot_id="adaptive_intervention_pilot_v3",
        required_shot_seed_domain=_SEED_DOMAIN,
        required_shots_per_rate=64,
        required_error_rates=_ERROR_RATES,
        code_distance=5,
        noise_model=_NOISE_MODEL,
        reference_modes=_REFERENCE_MODES,
        reference_probability_tolerance=probability_tolerance,
        reference_log_ratio_tolerance=log_ratio_tolerance,
        margin_threshold=threshold,
        positive_gain_threshold=positive_gain,
        oracle_tie_relative_tolerance=tie_tolerance,
        oracle_tie_absolute_tolerance=absolute_tie_tolerance,
        efficiency_unique_relative_separation=efficiency_separation,
        probe_action_ids=("columns_tol01", "rows_tol01"),
        actions=actions,
    )


def load_shot_config(path: Path, *, non_scientific_fixture: bool = False) -> ShotConfig:
    """Load the frozen scientific v3 or explicitly reserved two-shot test identity."""
    if type(non_scientific_fixture) is not bool:
        raise TypeError("non_scientific_fixture must be boolean")
    payload = _load_object(path)
    _exact_fields(payload, _SHOT_CONFIG_KEYS, "shot configuration")
    domain = payload["seed_domain"]
    required_domain = _FIXTURE_SEED_DOMAIN if non_scientific_fixture else _SEED_DOMAIN
    required_seed = _FIXTURE_CAMPAIGN_SEED if non_scientific_fixture else CAMPAIGN_SEED
    required_count = 1 if non_scientific_fixture else 64
    if not isinstance(domain, str) or domain != required_domain:
        raise ValueError(f"seed_domain must use the required domain: {required_domain}")
    expected_seed = int.from_bytes(hashlib.sha256(domain.encode()).digest()[:8], "big")
    if type(payload["campaign_seed"]) is not int or payload["campaign_seed"] != expected_seed:
        raise ValueError("campaign_seed must be the SHA-256 derivation of seed_domain")
    if expected_seed != required_seed:
        raise ValueError("campaign_seed does not match the frozen pilot seed")
    rates = payload["error_rates"]
    if (
        not isinstance(rates, list)
        or tuple(_finite_float(rate, "error_rates") for rate in rates) != _ERROR_RATES
    ):
        raise ValueError("error_rates must be exactly [0.1, 0.15]")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or type(payload["code_distance"]) is not int
        or payload["code_distance"] != 5
        or type(payload["shots_per_rate"]) is not int
        or payload["shots_per_rate"] != required_count
        or payload["noise_model"] != _NOISE_MODEL
    ):
        raise ValueError("shot configuration must match the frozen pilot identity")
    return ShotConfig(
        schema_version=1,
        seed_domain=domain,
        campaign_seed=expected_seed,
        code_distance=5,
        error_rates=_ERROR_RATES,
        shots_per_rate=required_count,
        noise_model=_NOISE_MODEL,
    )


def historical_domain_bindings(root: Path) -> dict[str, dict]:
    """Bind every declared non-pilot config domain, including future confirmations."""
    bindings = {}
    own_configs = {"adaptive_intervention_pilot.json", "adaptive_intervention_pilot_shots.json"}

    def domains(value):
        found = set()
        if isinstance(value, dict):
            for key, item in value.items():
                if key.endswith("seed_domain") and isinstance(item, str):
                    found.add(item)
                found.update(domains(item))
        elif isinstance(value, list):
            for item in value:
                found.update(domains(item))
        return found

    for path in sorted((root / "configs").rglob("*.json")):
        if path.parent == root / "configs" and path.name in own_configs:
            continue
        declared = domains(_load_object(path))
        if _SEED_DOMAIN in declared:
            raise ValueError(f"pilot/historical seed domain overlap: {path}")
        if declared:
            bindings[path.relative_to(root).as_posix()] = {
                "sha256": sha256_file(path),
                "domains": sorted(declared),
            }
    return bindings


def validate_pilot_shots(
    artifact_path: Path,
    shot_config_path: Path,
    *,
    non_scientific_fixture: bool = False,
) -> tuple[dict, list[dict]]:
    """Validate identity, provenance and exact physical-error/syndrome replay.

    Fixture mode accepts only index zero at each frozen rate in the reserved
    test domain. Neither mode accepts the other's domain or the retired v1/v2.
    """
    if type(non_scientific_fixture) is not bool:
        raise TypeError("non_scientific_fixture must be boolean")
    config = load_shot_config(shot_config_path, non_scientific_fixture=non_scientific_fixture)
    payload = _load_object(artifact_path)
    _exact_fields(payload, {"config", "provenance", "shots"}, "shot artifact")
    if json.dumps(payload["config"], sort_keys=True) != json.dumps(
        _load_object(shot_config_path),
        sort_keys=True,
    ):
        raise ValueError("shot artifact config does not match frozen config")
    rows = payload["shots"]
    per_rate = 1 if non_scientific_fixture else config.shots_per_rate
    if not isinstance(rows, list) or len(rows) != per_rate * len(config.error_rates):
        raise ValueError("shot count must match the exact per-rate requirement")
    provenance = payload["provenance"]
    if not isinstance(provenance, dict):
        raise TypeError("shot artifact provenance missing")
    _exact_fields(provenance, _SHOT_PROVENANCE_KEYS, "shot provenance")
    clean = _validate_shot_manifest(payload, _repository_root())
    if provenance["config_sha256"] != sha256_file(shot_config_path):
        raise ValueError("shot configuration hash mismatch")
    if not non_scientific_fixture and not clean:
        raise ValueError("scientific shot artifact requires clean committed provenance")
    historical_domain_bindings(_repository_root())
    code, model = PlanarCode(5, 5), DepolarizingErrorModel()
    seen, seeds, validated = set(), set(), []
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("malformed shot row")
        _exact_fields(
            row,
            {
                "shot_id",
                "shot_index",
                "sampler_seed",
                "error_rate",
                "error_bsf",
                "syndrome",
            },
            "shot row",
        )
        rate, index, seed = row["error_rate"], row["shot_index"], row["sampler_seed"]
        if (
            type(rate) not in (int, float)
            or rate not in config.error_rates
            or type(index) is not int
            or index not in range(per_rate)
            or type(seed) is not int
            or seed != _shot_seed(config.seed_domain, error_rate=rate, shot_index=index)
            or row["shot_id"] != f"d5/p{rate:.6f}/i{index:06d}"
            or (rate, index) in seen
            or seed in seeds
        ):
            raise ValueError("invalid or duplicate shot replay identity")
        error = _binary_vector(np.asarray(row["error_bsf"]), length=82, name="error")
        syndrome = _binary_vector(np.asarray(row["syndrome"]), length=40, name="syndrome")
        replayed = model.generate(code, rate, np.random.default_rng(seed))
        if not np.array_equal(error, replayed) or not np.array_equal(
            syndrome,
            pt.bsp(error, code.stabilizers.T),
        ):
            raise ValueError("physical error/syndrome replay mismatch")
        seen.add((rate, index))
        seeds.add(seed)
        validated.append(
            {
                "shot_id": row["shot_id"],
                "shot_index": index,
                "sampler_seed": seed,
                "error_rate": rate,
                "error": error,
                "syndrome": syndrome,
            }
        )
    if seen != {(rate, index) for rate in config.error_rates for index in range(per_rate)}:
        raise ValueError("shot count/index coverage mismatch")
    return payload, sorted(validated, key=lambda row: (row["error_rate"], row["shot_index"]))


def pilot_provenance(
    config_path: Path,
    shot_config_path: Path,
    shots_path: Path,
    *,
    scientific: bool,
) -> dict:
    """Bind producer, scientific sources, dependencies and immutable input bytes."""
    root = _repository_root()
    commit, dirty = _git_provenance(root)
    if scientific and dirty:
        raise ValueError("scientific runner requires clean committed provenance")
    sources = (
        *_SOURCE_LABELS,
        "src/qldpc_fno/decision/adaptive_intervention_pilot.py",
        "src/qldpc_fno/decision/adaptive_diagnostics.py",
        "experiments/35_run_adaptive_intervention_pilot.py",
    )
    provenance = {
        "git_commit": commit,
        "git_dirty": dirty,
        "source_sha256": {label: sha256_file(root / label) for label in sources},
        **_runtime_provenance(root),
        "inputs": {
            name: _file_binding(root, path, commit)
            for name, path in (
                ("config", config_path),
                ("shot_config", shot_config_path),
                ("shots", shots_path),
            )
        },
        "historical_domains": historical_domain_bindings(root),
    }
    if scientific:
        clean = _validate_producer(provenance, root, sources)
        committed_configs = all(
            provenance["inputs"][name]["committed"] for name in ("config", "shot_config")
        )
        committed_domains = all(
            _committed_digest(root, commit, label) == binding["sha256"]
            for label, binding in provenance["historical_domains"].items()
        )
        if not clean or not committed_configs or not committed_domains:
            raise ValueError("scientific runner requires clean committed configs and sources")
    # Raw shots live under ignored artifacts/; their clean producer and committed
    # generator config are validated separately. The input bytes are hash-bound.
    return provenance


def _write_pilot_json(path: Path, value: Mapping[str, object]) -> None:
    """Serialize strict canonical JSON completely before creating the output file."""
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def run_adaptive_intervention_pilot(
    config_path: Path,
    shots_path: Path,
    output_dir: Path,
    *,
    shot_config_path: Path | None = None,
    non_scientific_fixture: bool = False,
) -> dict:
    """Atomically publish deterministic results plus a separate host-timing sidecar."""
    started = perf_counter()
    if type(non_scientific_fixture) is not bool:
        raise TypeError("non_scientific_fixture must be boolean")
    config_path, shots_path, output_dir = map(Path, (config_path, shots_path, output_dir))
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    shot_config_path = (
        Path(shot_config_path)
        if shot_config_path
        else (
            _repository_root()
            / "configs"
            / (
                "adaptive_intervention_pilot_fixture_shots.json"
                if non_scientific_fixture
                else "adaptive_intervention_pilot_shots.json"
            )
        )
    )
    provenance = pilot_provenance(
        config_path,
        shot_config_path,
        shots_path,
        scientific=not non_scientific_fixture,
    )
    config = load_pilot_config(config_path)
    shot_payload, shots = validate_pilot_shots(
        shots_path,
        shot_config_path,
        non_scientific_fixture=non_scientific_fixture,
    )
    results, timed_shots = [], []
    for shot in shots:
        attempts = []
        results.append(evaluate_shot(shot, config, timing_records=attempts))
        timed_shots.append(
            {
                "shot_id": shot["shot_id"],
                "shot_index": shot["shot_index"],
                "error_rate": shot["error_rate"],
                "actions": attempts,
            }
        )
    summary = summarize_evaluations(results, config)
    if non_scientific_fixture:
        summary["advancement"].update(
            {
                "complete_pilot": False,
                "margin_gate_clause": False,
                "heterogeneous_efficiency_clause": False,
            }
        )
    if (
        pilot_provenance(
            config_path,
            shot_config_path,
            shots_path,
            scientific=not non_scientific_fixture,
        )
        != provenance
    ):
        raise ValueError("input/source provenance changed during evaluation")
    result = {
        "schema_version": 1,
        "status": "reduced_non_scientific" if non_scientific_fixture else "development_pilot",
        "scientific_eligible": not non_scientific_fixture,
        "provenance": provenance,
        "shot_provenance": shot_payload["provenance"],
        "config": asdict(config),
        "shots": results,
        "summary": summary,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as tmp:
        staging = Path(tmp) / "publication"
        raw_path = staging / "intervention_pilot.json"
        _write_pilot_json(raw_path, result)
        compact = {
            "schema_version": 1,
            "status": result["status"],
            "scientific_eligible": result["scientific_eligible"],
            "raw_sha256": sha256_file(raw_path),
            "raw_size_bytes": raw_path.stat().st_size,
            "provenance": provenance,
            "summary": summary,
        }
        _write_pilot_json(staging / "summary.json", compact)
        timing = {
            "schema_version": 1,
            "status": "nondeterministic_engineering_metadata",
            "measurement": "research_host_wall_time_not_decoder_latency",
            "host": {
                "node": platform.node(),
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            "raw_sha256": sha256_file(raw_path),
            "summary_sha256": sha256_file(staging / "summary.json"),
            "provenance": provenance,
            "shots": timed_shots,
            "total_host_wall_seconds": perf_counter() - started,
        }
        _write_pilot_json(staging / "timing.json", timing)
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
        staging.rename(output_dir)
    return result
