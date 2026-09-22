"""Frozen constants and strict configuration parsing for the planar confirmation."""

from __future__ import annotations

import json
import math
from pathlib import Path

SCIENTIFIC_DOMAIN = "qldpc-fno/simple-planar-confirmation/v1"
FIXTURE_DOMAIN = "qldpc-fno/simple-planar-confirmation/test-fixture/v1"
BOOTSTRAP_DOMAIN = "qldpc-fno/simple-planar-confirmation/bootstrap/v1"
SCIENTIFIC_CAMPAIGN_SEED = 2409828766515432030
FIXTURE_CAMPAIGN_SEED = 3697382327853009454
BOOTSTRAP_SEED = 2713269656809941213
MARGIN_THRESHOLD = 0.30710401263493464

_NOISE_MODEL = "qecsim_iid_depolarizing_code_capacity"
_SHOT_KEYS = {
    "schema_version",
    "seed_domain",
    "campaign_seed",
    "code_distance",
    "error_rates",
    "shots_per_rate",
    "noise_model",
}
_ACTION_KEYS = {"action_id", "mode", "chi", "tol"}

FROZEN_SHOT_CONFIG: dict[str, object] = {
    "schema_version": 1,
    "seed_domain": SCIENTIFIC_DOMAIN,
    "campaign_seed": SCIENTIFIC_CAMPAIGN_SEED,
    "code_distance": 5,
    "error_rates": [0.1, 0.15],
    "shots_per_rate": 2048,
    "noise_model": _NOISE_MODEL,
}
FIXTURE_SHOT_CONFIG: dict[str, object] = {
    **FROZEN_SHOT_CONFIG,
    "seed_domain": FIXTURE_DOMAIN,
    "campaign_seed": FIXTURE_CAMPAIGN_SEED,
    "shots_per_rate": 2,
}

_FROZEN_ACTIONS: list[dict[str, object]] = [
    {"action_id": "rows_tol003", "mode": "rows", "chi": None, "tol": 0.003},
    {"action_id": "columns_tol01", "mode": "columns", "chi": None, "tol": 0.01},
    {"action_id": "rows_tol01", "mode": "rows", "chi": None, "tol": 0.01},
    {"action_id": "columns_chi8", "mode": "columns", "chi": 8, "tol": None},
    {"action_id": "exact_columns", "mode": "columns", "chi": None, "tol": None},
    {"action_id": "exact_rows", "mode": "rows", "chi": None, "tol": None},
]

FROZEN_CONFIG: dict[str, object] = {
    "schema_version": 1,
    "contract_id": "simple_planar_confirmation_v1",
    "required_shot_seed_domain": SCIENTIFIC_DOMAIN,
    "fixture_seed_domain": FIXTURE_DOMAIN,
    "code_distance": 5,
    "noise_model": _NOISE_MODEL,
    "required_error_rates": [0.1, 0.15],
    "required_shots_per_rate": 2048,
    "policy_ids": ["fixed_rows_tol003", "margin_columns_chi8"],
    "comparator_id": "fixed_columns_chi8",
    "actions": _FROZEN_ACTIONS,
    "margin_threshold": MARGIN_THRESHOLD,
    "reference_probability_tolerance": 1e-10,
    "reference_log_ratio_tolerance": 1e-8,
    "reference_margin_discrepancy_factor": 2.0,
    "primary_endpoints": ["class_mismatch", "outcome_discordance"],
    "family_alpha": 0.05,
    "primary_alpha": 0.00625,
    "discrepancy_budget": 0.005,
    "bootstrap_domain": BOOTSTRAP_DOMAIN,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "bootstrap_replicates": 10_000,
    "bootstrap_generator": "PCG64",
    "bootstrap_quantile": 0.05,
    "bootstrap_quantile_method": "linear",
    "minimum_work_saving": 0.5,
    "order_rule": "lexicographic_permutations_index_mod_6",
    "invalidity_rule": "count_both_events_reference_blocks_positive",
    "status_rule": "immutable_pending_independent_replay",
}
_CONFIG_KEYS = set(FROZEN_CONFIG)


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            Path(path).read_text(),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid confirmation JSON") from error
    if not isinstance(value, dict):
        raise TypeError("confirmation configuration must be a JSON object")
    return value


def _same_literal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(  # type: ignore[arg-type]
            _same_literal(actual[key], value)  # type: ignore[index]
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(  # type: ignore[arg-type]
            _same_literal(left, right) for left, right in zip(actual, expected, strict=True)  # type: ignore[arg-type]
        )
    if isinstance(expected, float):
        return math.isfinite(actual) and actual == expected  # type: ignore[arg-type]
    return actual == expected


def _validate_actions(value: object) -> None:
    if not isinstance(value, list) or len(value) != len(_FROZEN_ACTIONS):
        raise ValueError("actions must contain exactly the six frozen actions")
    for item, expected in zip(value, _FROZEN_ACTIONS, strict=True):
        if not isinstance(item, dict) or set(item) != _ACTION_KEYS:
            raise ValueError("each action must contain exactly the frozen action fields")
        if not _same_literal(item, expected):
            raise ValueError("actions must use the frozen literal order and parameters")


def load_config(path: Path) -> dict[str, object]:
    """Load the exact frozen analysis configuration without scientific side effects."""
    payload = _load_object(path)
    if set(payload) != _CONFIG_KEYS:
        raise ValueError("analysis configuration must contain exactly the frozen fields")
    _validate_actions(payload["actions"])
    if not _same_literal(payload, FROZEN_CONFIG):
        raise ValueError("analysis configuration differs from the frozen contract")
    return payload


def load_shot_config(path: Path, *, fixture: bool) -> dict[str, object]:
    """Load exactly the production or explicitly reserved fixture shot identity."""
    if type(fixture) is not bool:
        raise TypeError("fixture must be boolean")
    payload = _load_object(path)
    if set(payload) != _SHOT_KEYS:
        raise ValueError("shot configuration must contain exactly the sampler fields")
    expected = FIXTURE_SHOT_CONFIG if fixture else FROZEN_SHOT_CONFIG
    if not _same_literal(payload, expected):
        mode = "fixture" if fixture else "scientific"
        raise ValueError(f"shot configuration differs from the frozen {mode} identity")
    return payload
