"""Frozen configuration identity for the adaptive intervention pilot."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from qldpc_fno.decision import adaptive_diagnostics as _diagnostics

_NOISE_MODEL = "qecsim_iid_depolarizing_code_capacity"
_SEED_DOMAIN = "qldpc-fno/adaptive-intervention-pilot/v1"
_ERROR_RATES = (0.1, 0.15)
_REFERENCE_MODES = ("columns", "rows")

CAMPAIGN_SEED = 16980117767564665917
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
    mean = 0.5 * (
        _diagnostics._probabilities(columns) + _diagnostics._probabilities(rows)
    )
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


def _selection(
    candidates: Sequence[CandidateOpportunity], *, efficiency: bool
) -> OracleSelection:
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
    runner_metrics = [float(metric(candidate)) for candidate in eligible if candidate != winner]
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
    if not all(_valid_selected_class(value) for value in (baseline_class, candidate_class, reference_class)):
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


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _load_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            Path(path).read_text(), object_pairs_hook=_reject_duplicate_json_keys
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
        or payload["pilot_id"] != "adaptive_intervention_pilot_v1"
    ):
        raise ValueError("unsupported pilot schema or identity")
    if payload["required_shot_seed_domain"] != _SEED_DOMAIN:
        raise ValueError("required_shot_seed_domain must use the frozen pilot domain")
    if type(payload["required_shots_per_rate"]) is not int or payload[
        "required_shots_per_rate"
    ] != 64:
        raise ValueError("required_shots_per_rate must be exactly 64")
    rates = payload["required_error_rates"]
    if not isinstance(rates, list) or tuple(_finite_float(rate, "required_error_rates") for rate in rates) != _ERROR_RATES:
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
        pilot_id="adaptive_intervention_pilot_v1",
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


def load_shot_config(path: Path) -> ShotConfig:
    """Load only the frozen 128-shot physical-error configuration."""
    payload = _load_object(path)
    _exact_fields(payload, _SHOT_CONFIG_KEYS, "shot configuration")
    domain = payload["seed_domain"]
    if not isinstance(domain, str) or domain != _SEED_DOMAIN:
        raise ValueError("seed_domain must use the frozen pilot domain")
    expected_seed = int.from_bytes(hashlib.sha256(domain.encode()).digest()[:8], "big")
    if type(payload["campaign_seed"]) is not int or payload["campaign_seed"] != expected_seed:
        raise ValueError("campaign_seed must be the SHA-256 derivation of seed_domain")
    if expected_seed != CAMPAIGN_SEED:
        raise ValueError("campaign_seed does not match the frozen pilot seed")
    rates = payload["error_rates"]
    if not isinstance(rates, list) or tuple(_finite_float(rate, "error_rates") for rate in rates) != _ERROR_RATES:
        raise ValueError("error_rates must be exactly [0.1, 0.15]")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or type(payload["code_distance"]) is not int
        or payload["code_distance"] != 5
        or type(payload["shots_per_rate"]) is not int
        or payload["shots_per_rate"] != 64
        or payload["noise_model"] != _NOISE_MODEL
    ):
        raise ValueError("shot configuration must match the frozen pilot identity")
    return ShotConfig(
        schema_version=1,
        seed_domain=domain,
        campaign_seed=expected_seed,
        code_distance=5,
        error_rates=_ERROR_RATES,
        shots_per_rate=64,
        noise_model=_NOISE_MODEL,
    )
