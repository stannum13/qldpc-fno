"""Frozen constants and strict configuration parsing for the planar confirmation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta

_SCIENTIFIC_DOMAIN_LITERAL = "qldpc-fno/simple-planar-confirmation/v1"
_FIXTURE_DOMAIN_LITERAL = "qldpc-fno/simple-planar-confirmation/test-fixture/v1"
_BOOTSTRAP_DOMAIN_LITERAL = "qldpc-fno/simple-planar-confirmation/bootstrap/v1"
_SCIENTIFIC_CAMPAIGN_SEED_LITERAL = 2409828766515432030
_FIXTURE_CAMPAIGN_SEED_LITERAL = 3697382327853009454
_BOOTSTRAP_SEED_LITERAL = 2713269656809941213
_MARGIN_THRESHOLD_LITERAL = 0.30710401263493464

# Public scalar snapshots are conveniences, never validator inputs.
SCIENTIFIC_DOMAIN = _SCIENTIFIC_DOMAIN_LITERAL
FIXTURE_DOMAIN = _FIXTURE_DOMAIN_LITERAL
BOOTSTRAP_DOMAIN = _BOOTSTRAP_DOMAIN_LITERAL
SCIENTIFIC_CAMPAIGN_SEED = _SCIENTIFIC_CAMPAIGN_SEED_LITERAL
FIXTURE_CAMPAIGN_SEED = _FIXTURE_CAMPAIGN_SEED_LITERAL
BOOTSTRAP_SEED = _BOOTSTRAP_SEED_LITERAL
MARGIN_THRESHOLD = _MARGIN_THRESHOLD_LITERAL

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
_ACTION_SPECS = (
    ("rows_tol003", "rows", None, 0.003),
    ("columns_tol01", "columns", None, 0.01),
    ("rows_tol01", "rows", None, 0.01),
    ("columns_chi8", "columns", 8, None),
    ("exact_columns", "columns", None, None),
    ("exact_rows", "rows", None, None),
)
_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "contract_id",
        "required_shot_seed_domain",
        "fixture_seed_domain",
        "code_distance",
        "noise_model",
        "required_error_rates",
        "required_shots_per_rate",
        "policy_ids",
        "comparator_id",
        "actions",
        "margin_threshold",
        "reference_probability_tolerance",
        "reference_log_ratio_tolerance",
        "reference_margin_discrepancy_factor",
        "primary_endpoints",
        "family_alpha",
        "primary_alpha",
        "discrepancy_budget",
        "bootstrap_domain",
        "bootstrap_seed",
        "bootstrap_replicates",
        "bootstrap_generator",
        "bootstrap_quantile",
        "bootstrap_quantile_method",
        "minimum_work_saving",
        "order_rule",
        "invalidity_rule",
        "status_rule",
    }
)


def _shot_config(*, fixture: bool) -> dict[str, object]:
    return {
        "schema_version": 1,
        "seed_domain": _FIXTURE_DOMAIN_LITERAL if fixture else _SCIENTIFIC_DOMAIN_LITERAL,
        "campaign_seed": (
            _FIXTURE_CAMPAIGN_SEED_LITERAL if fixture else _SCIENTIFIC_CAMPAIGN_SEED_LITERAL
        ),
        "code_distance": 5,
        "error_rates": [0.1, 0.15],
        "shots_per_rate": 2 if fixture else 2048,
        "noise_model": _NOISE_MODEL,
    }


def _frozen_actions() -> list[dict[str, object]]:
    return [
        {"action_id": action_id, "mode": mode, "chi": chi, "tol": tolerance}
        for action_id, mode, chi, tolerance in _ACTION_SPECS
    ]


def _analysis_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "contract_id": "simple_planar_confirmation_v1",
        "required_shot_seed_domain": _SCIENTIFIC_DOMAIN_LITERAL,
        "fixture_seed_domain": _FIXTURE_DOMAIN_LITERAL,
        "code_distance": 5,
        "noise_model": _NOISE_MODEL,
        "required_error_rates": [0.1, 0.15],
        "required_shots_per_rate": 2048,
        "policy_ids": ["fixed_rows_tol003", "margin_columns_chi8"],
        "comparator_id": "fixed_columns_chi8",
        "actions": _frozen_actions(),
        "margin_threshold": _MARGIN_THRESHOLD_LITERAL,
        "reference_probability_tolerance": 1e-10,
        "reference_log_ratio_tolerance": 1e-8,
        "reference_margin_discrepancy_factor": 2.0,
        "primary_endpoints": ["class_mismatch", "outcome_discordance"],
        "family_alpha": 0.05,
        "primary_alpha": 0.00625,
        "discrepancy_budget": 0.005,
        "bootstrap_domain": _BOOTSTRAP_DOMAIN_LITERAL,
        "bootstrap_seed": _BOOTSTRAP_SEED_LITERAL,
        "bootstrap_replicates": 10_000,
        "bootstrap_generator": "PCG64",
        "bootstrap_quantile": 0.05,
        "bootstrap_quantile_method": "linear",
        "minimum_work_saving": 0.5,
        "order_rule": "lexicographic_permutations_index_mod_6",
        "invalidity_rule": "count_both_events_reference_blocks_positive",
        "status_rule": "immutable_pending_independent_replay",
    }


# Public snapshots support ergonomic fixture construction. Validators deliberately
# rebuild their private canonical values, so callers cannot mutate the source of truth.
FROZEN_SHOT_CONFIG = _shot_config(fixture=False)
FIXTURE_SHOT_CONFIG = _shot_config(fixture=True)
FROZEN_CONFIG = _analysis_config()


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
    expected_actions = _frozen_actions()
    if not isinstance(value, list) or len(value) != len(expected_actions):
        raise ValueError("actions must contain exactly the six frozen actions")
    for item, expected in zip(value, expected_actions, strict=True):
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
    if not _same_literal(payload, _analysis_config()):
        raise ValueError("analysis configuration differs from the frozen contract")
    return payload


def load_shot_config(path: Path, *, fixture: bool) -> dict[str, object]:
    """Load exactly the production or explicitly reserved fixture shot identity."""
    if type(fixture) is not bool:
        raise TypeError("fixture must be boolean")
    payload = _load_object(path)
    if set(payload) != _SHOT_KEYS:
        raise ValueError("shot configuration must contain exactly the sampler fields")
    expected = _shot_config(fixture=fixture)
    if not _same_literal(payload, expected):
        mode = "fixture" if fixture else "scientific"
        raise ValueError(f"shot configuration differs from the frozen {mode} identity")
    return payload


def primary_upper(events: int, shots: int) -> float:
    """Return the frozen one-sided Clopper-Pearson upper confidence bound."""
    if type(events) is not int or type(shots) is not int:
        raise ValueError("events and shots must be integers")
    if shots <= 0 or not 0 <= events <= shots:
        raise ValueError("invalid binomial counts")
    if events == shots:
        return 1.0
    return float(beta.ppf(1.0 - 0.00625, events + 1, shots - events))


def _validate_endpoint_label(
    *, valid: bool, selected_class: int | None, failure: bool | None, name: str
) -> None:
    if valid:
        if type(selected_class) is not int or not 0 <= selected_class <= 3:
            raise ValueError(f"valid {name} class must be an integer in [0,3]")
        if type(failure) is not bool:
            raise ValueError(f"valid {name} failure must be boolean")
    elif selected_class is not None or failure is not None:
        raise ValueError(f"invalid {name} labels must be null")


def event_indicators(
    *,
    reference_valid: bool,
    policy_valid: bool,
    policy_class: int | None,
    reference_class: int | None,
    policy_failure: bool | None,
    reference_failure: bool | None,
) -> tuple[bool, bool]:
    """Return class-mismatch and outcome-discordance event indicators."""
    if type(reference_valid) is not bool or type(policy_valid) is not bool:
        raise ValueError("validity flags must be boolean")
    _validate_endpoint_label(
        valid=reference_valid,
        selected_class=reference_class,
        failure=reference_failure,
        name="reference",
    )
    _validate_endpoint_label(
        valid=policy_valid,
        selected_class=policy_class,
        failure=policy_failure,
        name="policy",
    )
    if not reference_valid or not policy_valid:
        return True, True
    return policy_class != reference_class, policy_failure != reference_failure


def _bootstrap_record(
    *,
    status: str,
    estimate: float | None,
    lower_bound: float | None,
    passed: bool,
    unavailable_reason: str | None,
) -> dict[str, object]:
    return {
        "status": status,
        "estimand": "equal_rate_mean_paired_relative_saving",
        "replicates": 10_000,
        "seed": _BOOTSTRAP_SEED_LITERAL,
        "bit_generator": "PCG64",
        "quantile_method": "linear",
        "estimate": estimate,
        "lower_bound": lower_bound,
        "threshold": 0.5,
        "passed": passed,
        "unavailable_reason": unavailable_reason,
    }


def _unavailable_bootstrap(reason: str) -> dict[str, object]:
    return _bootstrap_record(
        status="unavailable",
        estimate=None,
        lower_bound=None,
        passed=False,
        unavailable_reason=reason,
    )


def bootstrap_saving(paired: np.ndarray) -> dict[str, object]:
    """Bootstrap the frozen equal-rate mean paired relative-work saving."""
    if not isinstance(paired, np.ndarray):
        raise TypeError("paired data must be a NumPy array")
    fixture = paired.shape == (2, 2, 11)
    if not fixture and paired.shape != (2, 2048, 11):
        raise ValueError("paired data must have shape (2,2048,11) or fixture shape (2,2,11)")
    real_numeric = np.issubdtype(paired.dtype, np.integer) or np.issubdtype(
        paired.dtype, np.floating
    )
    if not real_numeric or np.issubdtype(paired.dtype, np.bool_):
        raise ValueError("paired data must have a real numeric non-boolean dtype")

    numeric = np.asarray(paired, dtype=np.float64)
    if not bool(np.all(np.isfinite(numeric))):
        return _unavailable_bootstrap("nonfinite_values")
    work = numeric[:, :, :3]
    if bool(np.any(work < 0.0)):
        return _unavailable_bootstrap("negative_work")
    if not bool(np.all(work == np.floor(work))):
        return _unavailable_bootstrap("noninteger_work")
    if bool(np.any(numeric[:, :, 1] <= 0.0)):
        return _unavailable_bootstrap("nonpositive_comparator_work")
    outcomes = numeric[:, :, 3:]
    if not bool(np.all((outcomes == 0.0) | (outcomes == 1.0))):
        return _unavailable_bootstrap("nonbinary_outcomes")
    if fixture:
        return _bootstrap_record(
            status="fixture_only",
            estimate=None,
            lower_bound=None,
            passed=False,
            unavailable_reason="fixture_analysis_is_noninferential",
        )

    rng = np.random.Generator(np.random.PCG64(_BOOTSTRAP_SEED_LITERAL))
    replicates = np.empty(10_000, dtype=np.float64)
    for replicate in range(10_000):
        means: list[float] = []
        for stratum in range(2):
            indices = rng.integers(0, 2048, size=2048)
            selected = numeric[stratum, indices, :]
            means.append(float(np.mean(1.0 - selected[:, 0] / selected[:, 1])))
        replicates[replicate] = (means[0] + means[1]) / 2.0
    estimate = float(np.mean(1.0 - numeric[:, :, 0] / numeric[:, :, 1]))
    lower = float(np.quantile(replicates, 0.05, method="linear"))
    return _bootstrap_record(
        status="available",
        estimate=estimate,
        lower_bound=lower,
        passed=lower > 0.5,
        unavailable_reason=None,
    )
