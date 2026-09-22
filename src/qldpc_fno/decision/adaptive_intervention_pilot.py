"""Frozen configuration identity for the adaptive intervention pilot."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

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


def _load_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text())
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
    if payload["required_shots_per_rate"] != 64:
        raise ValueError("required_shots_per_rate must be exactly 64")
    rates = payload["required_error_rates"]
    if not isinstance(rates, list) or tuple(_finite_float(rate, "required_error_rates") for rate in rates) != _ERROR_RATES:
        raise ValueError("required_error_rates must be exactly [0.1, 0.15]")
    if payload["code_distance"] != 5 or payload["noise_model"] != _NOISE_MODEL:
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
        or payload["code_distance"] != 5
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
