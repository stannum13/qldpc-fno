"""Frozen constants and strict configuration parsing for the planar confirmation."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
from qecsim import paulitools as pt
from qecsim.models.planar import PlanarCode
from scipy.stats import beta

from qldpc_fno.decision.planar_shot_accuracy import (
    certify_reference,
    logical_class_recovery,
    score_recovery,
)

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

POLICY_IDS = ("fixed_rows_tol003", "margin_columns_chi8", "fixed_columns_chi8")
_ARM_ORDERS = tuple(itertools.permutations(POLICY_IDS))

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

_EVALUATION_KEYS = {
    "shot_id", "error_rate", "shot_index", "sampler_seed", "error_bsf", "syndrome",
    "arm_order", "reference_order", "actions", "policies", "reference",
}
_RESULT_ACTION_KEYS = {
    "invocation_id", "action_id", "mode", "chi", "tol", "valid", "exception_type",
    "exception_message", "masses", "probabilities", "selected_class", "work",
    "estimated_arithmetic_flops",
}
_POLICY_KEYS = {
    "policy_id", "action_invocation_ids", "gate_accepted", "selected_class",
    "recovery_bsf", "syndrome_valid", "logical_signature", "physical_failure",
    "valid_recovery", "class_mismatch", "outcome_discordance",
    "estimated_arithmetic_flops",
}
_REFERENCE_KEYS = {
    "action_invocation_ids", "certified", "certificate", "exception_type",
    "exception_message", "probabilities", "selected_class", "recovery_bsf",
    "syndrome_valid", "logical_signature", "physical_failure",
    "estimated_arithmetic_flops",
}
_CERTIFICATE_KEYS = {
    "selected_class", "maximum_probability_discrepancy", "maximum_log_ratio_discrepancy",
    "probability_margins",
}
_WORK_KEYS = {
    "pairwise_contractions", "truncation_calls", "svd_calls", "qr_calls", "einsum_calls",
    "einsum_estimated_flops", "pairwise_output_elements", "decomposition_input_elements",
    "estimated_dense_decomposition_flops", "estimated_arithmetic_flops",
    "decomposition_attempts", "peak_observed_array_elements", "truncation_events",
    "contraction_sweeps", "terminal_residual_estimated_arithmetic_flops",
}
_ATTEMPT_KEYS = {
    "decomposition_kind", "matrix_rows", "matrix_columns", "estimated_flops", "succeeded",
    "exception_type",
}
_TRUNCATION_KEYS = {
    "index", "requested_chi", "requested_tol", "sweep_index", "input_bond_dimension",
    "input_elements", "spectral_summaries", "output_bond_dimension", "output_elements",
    "normalization_factor", "cumulative_estimated_arithmetic_flops",
}
_SPECTRUM_KEYS = {
    "matrix_rows", "matrix_columns", "singular_value_count", "retained_rank",
    "requested_chi", "requested_tol", "discarded_squared_weight_fraction",
    "spectral_entropy",
}
_SWEEP_KEYS = {
    "index", "network_rows", "network_columns", "requested_chi", "requested_tol",
    "pairwise_contractions", "truncation_event_start", "truncation_event_stop",
    "estimated_arithmetic_flops", "label",
}
_FAILURE_KEYS = {"exception_type", "message"}
_SELECTED_POLICIES = ("fixed_rows_tol003", "margin_columns_chi8")
_ENDPOINTS = ("class_mismatch", "outcome_discordance")
_RATES = (0.1, 0.15)
_INVOCATIONS = (
    "fixed_rows_tol003/rows_tol003",
    "margin_columns_chi8/columns_tol01",
    "margin_columns_chi8/rows_tol01",
    "margin_columns_chi8/columns_chi8",
    "fixed_columns_chi8/columns_chi8",
    "reference/exact_columns",
    "reference/exact_rows",
)
_NUMERICAL_ACTION_EXCEPTIONS = {"InvalidCosetMassError", "InvalidContractionError"}


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


def arm_order(shot_index: int) -> tuple[str, str, str]:
    """Return the frozen counterbalanced policy order for one within-rate index."""
    if type(shot_index) is not int or shot_index < 0:
        raise ValueError("invalid shot index")
    return _ARM_ORDERS[shot_index % len(_ARM_ORDERS)]


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


def _exact_fields(value: object, fields: set[str], name: str) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    if set(value) != fields:
        raise ValueError(f"{name} must contain exactly its frozen fields")
    return value


def _strict_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_number(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


def _nullable_string(value: object, name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string or null")
    return value


def _binary_list(value: object, length: int, name: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(type(item) is not int or item not in (0, 1) for item in value)
    ):
        raise ValueError(f"{name} must be a binary vector of length {length}")
    return value


def _probabilities(value: object, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{name} must contain four probabilities")
    result = [_finite_number(item, name) for item in value]
    if any(item <= 0 for item in result) or not math.isclose(
        sum(result), 1.0, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(f"{name} must be a normalized positive distribution")
    return result


def _nullable_positive(value: object, name: str) -> float | None:
    if value is None:
        return None
    result = _finite_number(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive or null")
    return result


def _decomposition_cost(kind: str, rows: int, columns: int) -> int:
    rank = min(rows, columns)
    if kind == "svd":
        return int(4 * rows * columns * rank + 8 * rank**3)
    if kind == "qr":
        return int(2 * rows * columns * rank - 2 * rank**3 / 3)
    raise ValueError("unknown decomposition kind")


def _validate_work(value: object) -> int | None:
    if value is None:
        return None
    work = _exact_fields(value, _WORK_KEYS, "work")
    integer_fields = _WORK_KEYS - {"decomposition_attempts", "truncation_events", "contraction_sweeps"}
    for field in integer_fields:
        _strict_int(work[field], f"work.{field}")
    attempts = work["decomposition_attempts"]
    if not isinstance(attempts, list):
        raise TypeError("work.decomposition_attempts must be a list")
    dense = 0
    decomposition_elements = 0
    observed_element_lower_bound = 0
    counts = {"svd": 0, "qr": 0}
    for raw in attempts:
        attempt = _exact_fields(raw, _ATTEMPT_KEYS, "decomposition attempt")
        kind = attempt["decomposition_kind"]
        if kind not in counts:
            raise ValueError("invalid decomposition kind")
        rows = _strict_int(attempt["matrix_rows"], "matrix_rows", minimum=1)
        columns = _strict_int(attempt["matrix_columns"], "matrix_columns", minimum=1)
        expected = _decomposition_cost(kind, rows, columns)
        if attempt["estimated_flops"] != expected:
            raise ValueError("decomposition charge does not reconcile")
        succeeded = _boolean(attempt["succeeded"], "attempt.succeeded")
        exception = _nullable_string(attempt["exception_type"], "attempt.exception_type")
        if succeeded != (exception is None):
            raise ValueError("decomposition success and exception disagree")
        counts[kind] += 1
        dense += expected
        decomposition_elements += rows * columns
        observed_element_lower_bound = max(observed_element_lower_bound, rows * columns)
    if work["svd_calls"] != counts["svd"] or work["qr_calls"] != counts["qr"]:
        raise ValueError("decomposition counters do not reconcile")
    if work["estimated_dense_decomposition_flops"] != dense:
        raise ValueError("dense decomposition work does not reconcile")
    if work["decomposition_input_elements"] != decomposition_elements:
        raise ValueError("decomposition input elements do not reconcile")
    total = work["einsum_estimated_flops"] + dense
    if work["estimated_arithmetic_flops"] != total:
        raise ValueError("total action work does not reconcile")
    truncations = work["truncation_events"]
    if not isinstance(truncations, list):
        raise TypeError("work.truncation_events must be a list")
    for raw in truncations:
        event = _exact_fields(raw, _TRUNCATION_KEYS, "truncation event")
        for field in (
            "index", "input_bond_dimension", "input_elements", "output_bond_dimension",
            "output_elements", "cumulative_estimated_arithmetic_flops",
        ):
            _strict_int(event[field], f"truncation.{field}")
        observed_element_lower_bound = max(
            observed_element_lower_bound,
            event["input_elements"],
            event["output_elements"],
        )
        if event["sweep_index"] is not None:
            _strict_int(event["sweep_index"], "truncation.sweep_index")
        if event["requested_chi"] is not None:
            _strict_int(event["requested_chi"], "truncation.requested_chi", minimum=1)
        _nullable_positive(event["requested_tol"], "truncation.requested_tol")
        _finite_number(event["normalization_factor"], "normalization_factor")
        spectra = event["spectral_summaries"]
        if not isinstance(spectra, list):
            raise TypeError("spectral_summaries must be a list")
        for raw_spectrum in spectra:
            spectrum = _exact_fields(raw_spectrum, _SPECTRUM_KEYS, "spectral summary")
            for field in (
                "matrix_rows", "matrix_columns", "singular_value_count", "retained_rank",
            ):
                _strict_int(spectrum[field], f"spectrum.{field}")
            observed_element_lower_bound = max(
                observed_element_lower_bound,
                spectrum["matrix_rows"] * spectrum["matrix_columns"],
            )
            if spectrum["requested_chi"] is not None:
                _strict_int(spectrum["requested_chi"], "spectrum.requested_chi", minimum=1)
            _nullable_positive(spectrum["requested_tol"], "spectrum.requested_tol")
            discarded = _finite_number(
                spectrum["discarded_squared_weight_fraction"], "discarded fraction"
            )
            entropy = _finite_number(spectrum["spectral_entropy"], "spectral entropy")
            if not 0 <= discarded <= 1 or entropy < 0:
                raise ValueError("invalid spectral summary")
    if work["truncation_calls"] != len(truncations):
        raise ValueError("truncation call count does not reconcile")
    sweeps = work["contraction_sweeps"]
    if not isinstance(sweeps, list):
        raise TypeError("work.contraction_sweeps must be a list")
    sweep_work = 0
    sweep_pairwise = 0
    for raw in sweeps:
        if not isinstance(raw, dict):
            raise TypeError("contraction sweep must be an object")
        allowed = _SWEEP_KEYS | ({"failure"} if "failure" in raw else set())
        sweep = _exact_fields(raw, allowed, "contraction sweep")
        for field in (
            "index", "network_rows", "network_columns", "pairwise_contractions",
            "truncation_event_start", "truncation_event_stop", "estimated_arithmetic_flops",
        ):
            _strict_int(sweep[field], f"sweep.{field}")
        if sweep["requested_chi"] is not None:
            _strict_int(sweep["requested_chi"], "sweep.requested_chi", minimum=1)
        _nullable_positive(sweep["requested_tol"], "sweep.requested_tol")
        if not isinstance(sweep["label"], str):
            raise TypeError("sweep label must be a string")
        if "failure" in sweep:
            failure = _exact_fields(sweep["failure"], _FAILURE_KEYS, "sweep failure")
            if not all(isinstance(failure[field], str) for field in _FAILURE_KEYS):
                raise ValueError("sweep failure fields must be strings")
        sweep_work += sweep["estimated_arithmetic_flops"]
        sweep_pairwise += sweep["pairwise_contractions"]
        if not 0 <= sweep["truncation_event_start"] <= sweep["truncation_event_stop"] <= len(
            truncations
        ):
            raise ValueError("sweep truncation range does not reconcile")
    if sweep_work + work["terminal_residual_estimated_arithmetic_flops"] != total:
        raise ValueError("sweep work does not reconcile")
    if sweep_pairwise > work["pairwise_contractions"]:
        raise ValueError("pairwise contraction count does not reconcile")
    if (work["einsum_calls"] == 0) != (work["einsum_estimated_flops"] == 0):
        raise ValueError("einsum trace counters do not reconcile")
    if work["einsum_estimated_flops"] < work["einsum_calls"]:
        raise ValueError("einsum trace work is smaller than its call count")
    if (work["pairwise_contractions"] == 0) != (work["pairwise_output_elements"] == 0):
        raise ValueError("pairwise trace counters do not reconcile")
    if work["pairwise_output_elements"] < work["pairwise_contractions"]:
        raise ValueError("pairwise output trace is smaller than its call count")
    activity = (
        work["einsum_calls"]
        + work["pairwise_contractions"]
        + work["svd_calls"]
        + work["qr_calls"]
    )
    if activity > 0:
        observed_element_lower_bound = max(observed_element_lower_bound, 1)
    if work["peak_observed_array_elements"] < observed_element_lower_bound:
        raise ValueError("peak observed array trace does not reconcile")
    return total


def _action_spec(action_id: str) -> tuple[str, int | None, float | None]:
    specs = {action[0]: (action[1], action[2], action[3]) for action in _ACTION_SPECS}
    if action_id not in specs:
        raise ValueError("unknown action_id")
    return specs[action_id]


def _validate_action(value: object, expected_invocation: str) -> dict:
    action = _exact_fields(value, _RESULT_ACTION_KEYS, "action")
    if action["invocation_id"] != expected_invocation:
        raise ValueError("action invocation identity or order changed")
    action_id = expected_invocation.split("/", 1)[1]
    if action["action_id"] != action_id:
        raise ValueError("action_id does not match invocation")
    mode, chi, tolerance = _action_spec(action_id)
    if (
        type(action["mode"]) is not type(mode)
        or type(action["chi"]) is not type(chi)
        or type(action["tol"]) is not type(tolerance)
        or (action["mode"], action["chi"], action["tol"]) != (mode, chi, tolerance)
    ):
        raise ValueError("action parameters differ from frozen specification")
    valid = _boolean(action["valid"], "action.valid")
    exception_type = _nullable_string(action["exception_type"], "action.exception_type")
    _nullable_string(action["exception_message"], "action.exception_message")
    work_total = _validate_work(action["work"])
    recorded_work = action["estimated_arithmetic_flops"]
    if recorded_work is not None:
        _strict_int(recorded_work, "action.estimated_arithmetic_flops")
    if work_total != recorded_work:
        raise ValueError("action work total differs from work trace")
    if valid:
        if exception_type is not None or action["exception_message"] is not None:
            raise ValueError("valid action cannot contain an exception")
        masses = action["masses"]
        if not isinstance(masses, list) or len(masses) != 4:
            raise ValueError("valid action masses must contain four entries")
        mass_values = [_finite_number(item, "action mass") for item in masses]
        if any(item <= 0 for item in mass_values):
            raise ValueError("valid action masses must be positive")
        probabilities = _probabilities(action["probabilities"], "action probabilities")
        normalized_masses = np.asarray(mass_values, dtype=float) / sum(mass_values)
        if not np.allclose(probabilities, normalized_masses, rtol=1e-12, atol=1e-15):
            raise ValueError("action probabilities do not normalize recorded masses")
        selected = _strict_int(action["selected_class"], "action.selected_class")
        if selected > 3 or selected != int(np.argmax(probabilities)):
            raise ValueError("action selected class is inconsistent")
    else:
        if exception_type is None or not isinstance(action["exception_message"], str):
            raise ValueError("invalid action requires exception details")
        if exception_type not in _NUMERICAL_ACTION_EXCEPTIONS:
            raise ValueError("invalid action contains an undeclared exception type")
        if action["probabilities"] is not None or action["selected_class"] is not None:
            raise ValueError("invalid action cannot retain a posterior decision")
        if action["masses"] is not None:
            if not isinstance(action["masses"], list) or len(action["masses"]) != 4:
                raise ValueError("invalid finite masses must have length four")
            for mass in action["masses"]:
                _finite_number(mass, "invalid action mass")
    return action


def _margin_accepts(actions: list[dict]) -> bool:
    columns, rows = actions[:2]
    if not columns["valid"] or not rows["valid"]:
        return False
    if columns["selected_class"] != rows["selected_class"]:
        return False
    margins = []
    for action in (columns, rows):
        ordered = sorted(action["probabilities"])
        margins.append(ordered[-1] - ordered[-2])
    return min(margins) > _MARGIN_THRESHOLD_LITERAL


def _validate_score(
    *,
    code: PlanarCode,
    error: np.ndarray,
    syndrome: np.ndarray,
    recovery: object,
    syndrome_valid: object,
    signature: object,
    physical_failure: object,
    valid: bool,
    selected_class: object,
    name: str,
) -> None:
    _boolean(syndrome_valid, f"{name}.syndrome_valid")
    if not valid:
        if physical_failure is not True or syndrome_valid is not False:
            raise ValueError(f"invalid {name} must be a physical failure")
        if signature is not None:
            raise ValueError(f"invalid {name} logical signature must be null")
        if recovery is None:
            return
        recovery_values = _binary_list(recovery, 82, f"{name}.recovery_bsf")
        score = score_recovery(code, error, syndrome, np.asarray(recovery_values, dtype=np.uint8))
        if score["syndrome_valid"]:
            raise ValueError(f"invalid {name} retains a syndrome-valid recovery")
        return
    recovery_values = _binary_list(recovery, 82, f"{name}.recovery_bsf")
    if type(selected_class) is not int or not 0 <= selected_class <= 3:
        raise ValueError(f"valid {name} requires a logical class")
    recovery_array = np.asarray(recovery_values, dtype=np.uint8)
    canonical = logical_class_recovery(code, syndrome, selected_class)
    difference = recovery_array ^ canonical
    difference_syndrome = np.asarray(pt.bsp(difference, code.stabilizers.T), dtype=np.uint8)
    difference_logical = np.asarray(pt.bsp(difference, code.logicals.T), dtype=np.uint8)
    if np.any(difference_syndrome) or np.any(difference_logical):
        raise ValueError(f"{name} recovery is not a stabilizer-equivalent class representative")
    score = score_recovery(code, error, syndrome, recovery_array)
    expected_signature = score["logical_signature"].tolist()
    expected_failure = not bool(score["syndrome_valid"]) or bool(score["logical_failure"])
    if (
        syndrome_valid is not bool(score["syndrome_valid"])
        or signature != expected_signature
        or physical_failure is not expected_failure
    ):
        raise ValueError(f"{name} logical or physical score is inconsistent")


def _validate_policy(
    value: object,
    *,
    policy_id: str,
    actions: list[dict],
    reference: dict,
    code: PlanarCode,
    error: np.ndarray,
    syndrome: np.ndarray,
) -> dict:
    policy = _exact_fields(value, _POLICY_KEYS, "policy")
    if policy["policy_id"] != policy_id:
        raise ValueError("policy identity or order changed")
    invocation_ids = [action["invocation_id"] for action in actions]
    if policy["action_invocation_ids"] != invocation_ids:
        raise ValueError("policy action references do not resolve exactly")
    accepted = _margin_accepts(actions) if policy_id == "margin_columns_chi8" else None
    if policy["gate_accepted"] is not accepted:
        raise ValueError("policy gate result is inconsistent")
    selected_action = actions[0] if accepted is not False else actions[-1]
    expected_class = selected_action["selected_class"] if selected_action["valid"] else None
    if type(policy["selected_class"]) is not type(expected_class) or policy["selected_class"] != expected_class:
        raise ValueError("policy selected class is inconsistent")
    valid_recovery = _boolean(policy["valid_recovery"], "policy.valid_recovery")
    expected_work = (
        None
        if any(action["estimated_arithmetic_flops"] is None for action in actions)
        else sum(action["estimated_arithmetic_flops"] for action in actions)
    )
    if policy["estimated_arithmetic_flops"] != expected_work:
        raise ValueError("policy work does not equal its action sum")
    _validate_score(
        code=code,
        error=error,
        syndrome=syndrome,
        recovery=policy["recovery_bsf"],
        syndrome_valid=policy["syndrome_valid"],
        signature=policy["logical_signature"],
        physical_failure=policy["physical_failure"],
        valid=valid_recovery,
        selected_class=policy["selected_class"],
        name="policy",
    )
    expected_events = event_indicators(
        reference_valid=reference["certified"],
        policy_valid=valid_recovery,
        policy_class=policy["selected_class"] if valid_recovery else None,
        reference_class=reference["selected_class"],
        policy_failure=policy["physical_failure"] if valid_recovery else None,
        reference_failure=reference["physical_failure"],
    )
    recorded_events = (
        _boolean(policy["class_mismatch"], "policy.class_mismatch"),
        _boolean(policy["outcome_discordance"], "policy.outcome_discordance"),
    )
    if recorded_events != expected_events:
        raise ValueError("persisted primary event flags are inconsistent")
    return policy


def _validate_reference(
    value: object,
    *,
    actions: list[dict],
    code: PlanarCode,
    error: np.ndarray,
    syndrome: np.ndarray,
) -> dict:
    reference = _exact_fields(value, _REFERENCE_KEYS, "reference")
    if reference["action_invocation_ids"] != [action["invocation_id"] for action in actions]:
        raise ValueError("reference action references do not resolve exactly")
    certified = _boolean(reference["certified"], "reference.certified")
    expected_work = (
        None
        if any(action["estimated_arithmetic_flops"] is None for action in actions)
        else sum(action["estimated_arithmetic_flops"] for action in actions)
    )
    if reference["estimated_arithmetic_flops"] != expected_work:
        raise ValueError("reference work does not equal its action sum")
    numerical: dict[str, object] | None = None
    if all(action["valid"] for action in actions):
        by_mode = {action["mode"]: action for action in actions}
        try:
            numerical = certify_reference(
                np.asarray(by_mode["columns"]["masses"], dtype=float),
                np.asarray(by_mode["rows"]["masses"], dtype=float),
            )
        except ValueError:
            numerical = None
    recorded_certificate = reference["certificate"]
    if numerical is None:
        if recorded_certificate is not None:
            raise ValueError("reference certificate is inconsistent with exact views")
    else:
        recorded = _exact_fields(recorded_certificate, _CERTIFICATE_KEYS, "certificate")
        recorded_class = _strict_int(recorded["selected_class"], "certificate.selected_class")
        if recorded_class > 3 or recorded_class != numerical["selected_class"]:
            raise ValueError("reference certificate selected class is inconsistent")
        for field in ("maximum_probability_discrepancy", "maximum_log_ratio_discrepancy"):
            if not math.isclose(
                _finite_number(recorded[field], f"certificate.{field}"),
                float(numerical[field]),
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ValueError("reference certificate is inconsistent with exact views")
        margins = recorded["probability_margins"]
        if not isinstance(margins, list) or len(margins) != 2 or not np.allclose(
            margins, numerical["probability_margins"], rtol=1e-12, atol=1e-15
        ):
            raise ValueError("reference certificate margins are inconsistent")
    if certified:
        if (
            numerical is None
            or type(reference["selected_class"]) is not int
            or reference["selected_class"] != numerical["selected_class"]
        ):
            raise ValueError("certified reference lacks its numerical certificate")
        probabilities = [
            np.asarray(action["probabilities"], dtype=float) for action in actions
        ]
        target = np.mean(probabilities, axis=0)
        target /= target.sum()
        recorded = _probabilities(reference["probabilities"], "reference.probabilities")
        if not np.allclose(recorded, target, rtol=1e-12, atol=1e-12):
            raise ValueError("reference posterior is inconsistent")
        if reference["exception_type"] is not None or reference["exception_message"] is not None:
            raise ValueError("certified reference cannot contain an exception")
        _validate_score(
            code=code,
            error=error,
            syndrome=syndrome,
            recovery=reference["recovery_bsf"],
            syndrome_valid=reference["syndrome_valid"],
            signature=reference["logical_signature"],
            physical_failure=reference["physical_failure"],
            valid=True,
            selected_class=reference["selected_class"],
            name="reference",
        )
    else:
        if reference["selected_class"] is not None or reference["probabilities"] is not None:
            raise ValueError("uncertified reference labels must be null")
        if reference["physical_failure"] is not None or reference["logical_signature"] is not None:
            raise ValueError("uncertified reference outcomes must be null")
        if reference["syndrome_valid"] is not False:
            raise ValueError("uncertified reference must not claim syndrome validity")
        if numerical is None:
            if reference["recovery_bsf"] is not None:
                raise ValueError("uncertified numerical reference recovery must be null")
        else:
            recovery = _binary_list(reference["recovery_bsf"], 82, "reference.recovery_bsf")
            score = score_recovery(code, error, syndrome, np.asarray(recovery, dtype=np.uint8))
            if score["syndrome_valid"]:
                raise ValueError("uncertified reference recovery is unexpectedly valid")
        if not isinstance(reference["exception_type"], str) or not isinstance(
            reference["exception_message"], str
        ):
            raise ValueError("uncertified reference requires exception details")
    return reference


def validate_result_rows(rows: list[dict], *, fixture: bool) -> None:
    """Validate complete joined evaluation rows without deriving any shot seed."""
    if type(fixture) is not bool:
        raise TypeError("fixture must be boolean")
    if not isinstance(rows, list):
        raise TypeError("result rows must be a list")
    per_rate = 2 if fixture else 2048
    if len(rows) != 2 * per_rate:
        raise ValueError("result rows must contain the complete frozen sample")
    code = PlanarCode(5, 5)
    seen: set[tuple[float, int]] = set()
    seeds: set[int] = set()
    for position, raw in enumerate(rows):
        row = _exact_fields(raw, _EVALUATION_KEYS, "evaluation row")
        expected_rate = _RATES[position // per_rate]
        expected_index = position % per_rate
        rate = _finite_number(row["error_rate"], "error_rate")
        index = _strict_int(row["shot_index"], "shot_index")
        if rate != expected_rate or index != expected_index or (rate, index) in seen:
            raise ValueError("result rows are not canonical complete rate/index membership")
        expected_id = f"d5/p{rate:.6f}/i{index:06d}"
        seed = _strict_int(row["sampler_seed"], "sampler_seed")
        if row["shot_id"] != expected_id or seed in seeds:
            raise ValueError("invalid or duplicate shot identity")
        seen.add((rate, index))
        seeds.add(seed)
        error = np.asarray(_binary_list(row["error_bsf"], 82, "error_bsf"), dtype=np.uint8)
        syndrome = np.asarray(_binary_list(row["syndrome"], 40, "syndrome"), dtype=np.uint8)
        if not np.array_equal(np.asarray(pt.bsp(error, code.stabilizers.T), dtype=np.uint8), syndrome):
            raise ValueError("physical error and syndrome are not joined")
        order = list(arm_order(index))
        reference_order = ["columns", "rows"] if index % 2 == 0 else ["rows", "columns"]
        if row["arm_order"] != order or row["reference_order"] != reference_order:
            raise ValueError("execution order differs from the frozen schedule")
        if not isinstance(row["policies"], list) or len(row["policies"]) != 3:
            raise ValueError("each row must contain all three policies")
        policies_by_id = {policy["policy_id"]: policy for policy in row["policies"] if isinstance(policy, dict) and "policy_id" in policy}
        if list(policies_by_id) != order or len(policies_by_id) != 3:
            raise ValueError("policy rows must follow the arm order exactly")
        expected_invocations: list[str] = []
        policy_invocations: dict[str, list[str]] = {}
        for policy_id in order:
            if policy_id == "fixed_rows_tol003":
                invocations = ["fixed_rows_tol003/rows_tol003"]
            elif policy_id == "fixed_columns_chi8":
                invocations = ["fixed_columns_chi8/columns_chi8"]
            else:
                policy = policies_by_id[policy_id]
                accepted = policy.get("gate_accepted")
                invocations = [
                    "margin_columns_chi8/columns_tol01",
                    "margin_columns_chi8/rows_tol01",
                ]
                if accepted is False:
                    invocations.append("margin_columns_chi8/columns_chi8")
                elif accepted is not True:
                    raise ValueError("margin gate must be boolean")
            policy_invocations[policy_id] = invocations
            expected_invocations.extend(invocations)
        reference_invocations = [f"reference/exact_{mode}" for mode in reference_order]
        expected_invocations.extend(reference_invocations)
        if not isinstance(row["actions"], list) or len(row["actions"]) != len(expected_invocations):
            raise ValueError("missing or extra action row")
        actions = [
            _validate_action(action, invocation)
            for action, invocation in zip(row["actions"], expected_invocations, strict=True)
        ]
        action_map = {action["invocation_id"]: action for action in actions}
        if len(action_map) != len(actions):
            raise ValueError("duplicate action invocation")
        reference_actions = [action_map[invocation] for invocation in reference_invocations]
        reference = _validate_reference(
            row["reference"], actions=reference_actions, code=code, error=error, syndrome=syndrome
        )
        for policy_id, raw_policy in zip(order, row["policies"], strict=True):
            _validate_policy(
                raw_policy,
                policy_id=policy_id,
                actions=[action_map[item] for item in policy_invocations[policy_id]],
                reference=reference,
                code=code,
                error=error,
                syndrome=syndrome,
            )


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _policy_map(row: dict) -> dict[str, dict]:
    return {policy["policy_id"]: policy for policy in row["policies"]}


def _work_ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _sum_nullable(values: list[int | None]) -> int | None:
    return None if any(value is None for value in values) else sum(values)  # type: ignore[arg-type]


def summarize(rows: list[dict], *, fixture: bool) -> dict[str, object]:
    """Construct the frozen deterministic summary from independently validated rows."""
    validate_result_rows(rows, fixture=fixture)
    per_rate = 2 if fixture else 2048
    strata = {rate: rows[offset * per_rate : (offset + 1) * per_rate] for offset, rate in enumerate(_RATES)}
    sample_counts = [
        {"error_rate": rate, "expected": per_rate, "observed": len(strata[rate]), "complete": len(strata[rate]) == per_rate}
        for rate in _RATES
    ]
    invalid_counts = []
    for rate in _RATES:
        rate_rows = strata[rate]
        invalid_actions = {identity: 0 for identity in _INVOCATIONS}
        unmetered_actions = {identity: 0 for identity in _INVOCATIONS}
        invalid_recoveries = {policy_id: 0 for policy_id in POLICY_IDS}
        for row in rate_rows:
            for action in row["actions"]:
                invalid_actions[action["invocation_id"]] += int(not action["valid"])
                unmetered_actions[action["invocation_id"]] += int(
                    action["work"] is None or action["estimated_arithmetic_flops"] is None
                )
            for policy in row["policies"]:
                invalid_recoveries[policy["policy_id"]] += int(not policy["valid_recovery"])
        invalid_counts.append(
            {
                "error_rate": rate,
                "reference_uncertified": sum(not row["reference"]["certified"] for row in rate_rows),
                "invalid_recoveries": invalid_recoveries,
                "invalid_actions": invalid_actions,
                "unmetered_actions": unmetered_actions,
            }
        )
    primary = []
    for policy_id in _SELECTED_POLICIES:
        for rate in _RATES:
            for endpoint in _ENDPOINTS:
                events = sum(_policy_map(row)[policy_id][endpoint] for row in strata[rate])
                upper = primary_upper(events, per_rate)
                primary.append(
                    {
                        "policy_id": policy_id,
                        "error_rate": rate,
                        "endpoint": endpoint,
                        "events": events,
                        "shots": per_rate,
                        "alpha": 0.00625,
                        "upper_bound": upper,
                        "threshold": 0.005,
                        "gate_passed": upper <= 0.005,
                    }
                )
    policies = []
    for policy_id in (*POLICY_IDS, "reference"):
        for rate in _RATES:
            rate_rows = strata[rate]
            if policy_id == "reference":
                valid = sum(row["reference"]["certified"] for row in rate_rows)
                failures = sum(
                    (not row["reference"]["certified"])
                    or bool(row["reference"]["physical_failure"])
                    for row in rate_rows
                )
            else:
                values = [_policy_map(row)[policy_id] for row in rate_rows]
                valid = sum(policy["valid_recovery"] for policy in values)
                failures = sum(policy["physical_failure"] for policy in values)
            policies.append(
                {
                    "policy_id": policy_id,
                    "error_rate": rate,
                    "shots": per_rate,
                    "valid_recoveries": valid,
                    "physical_failures": failures,
                    "physical_failure_rate": failures / per_rate,
                }
            )
    comparator = []
    for rate in _RATES:
        values = [_policy_map(row)["fixed_columns_chi8"] for row in strata[rate]]
        mismatches = sum(value["class_mismatch"] for value in values)
        discordances = sum(value["outcome_discordance"] for value in values)
        comparator.append(
            {
                "error_rate": rate,
                "shots": per_rate,
                "class_mismatches": mismatches,
                "outcome_discordances": discordances,
                "class_mismatch_rate": mismatches / per_rate,
                "outcome_discordance_rate": discordances / per_rate,
            }
        )
    coverage = []
    for rate in _RATES:
        margins = [_policy_map(row)["margin_columns_chi8"] for row in strata[rate]]
        accepted_values = [policy for policy in margins if policy["gate_accepted"]]
        accepted = len(accepted_values)
        mismatches = sum(policy["class_mismatch"] for policy in accepted_values)
        discordances = sum(policy["outcome_discordance"] for policy in accepted_values)
        coverage.append(
            {
                "error_rate": rate,
                "shots": per_rate,
                "accepted": accepted,
                "escalated": per_rate - accepted,
                "accepted_class_mismatches": mismatches,
                "accepted_outcome_discordances": discordances,
                "coverage": accepted / per_rate,
                "accepted_class_mismatch_rate": _safe_rate(mismatches, accepted),
                "accepted_outcome_discordance_rate": _safe_rate(discordances, accepted),
            }
        )
    pairs = (
        ("fixed_rows_tol003", "margin_columns_chi8"),
        ("fixed_rows_tol003", "fixed_columns_chi8"),
        ("margin_columns_chi8", "fixed_columns_chi8"),
        ("fixed_rows_tol003", "reference"),
        ("margin_columns_chi8", "reference"),
        ("fixed_columns_chi8", "reference"),
    )
    paired_physical = []
    for rate in _RATES:
        for left, right in pairs:
            counts = {"both_succeed": 0, "left_only_fails": 0, "right_only_fails": 0, "both_fail": 0, "uncertified_reference": 0}
            for row in strata[rate]:
                policy_values = _policy_map(row)
                left_failure = bool(policy_values[left]["physical_failure"])
                if right == "reference":
                    if not row["reference"]["certified"]:
                        counts["uncertified_reference"] += 1
                        continue
                    right_failure = bool(row["reference"]["physical_failure"])
                else:
                    right_failure = bool(policy_values[right]["physical_failure"])
                key = (
                    "both_fail" if left_failure and right_failure
                    else "left_only_fails" if left_failure
                    else "right_only_fails" if right_failure
                    else "both_succeed"
                )
                counts[key] += 1
            paired_physical.append(
                {"error_rate": rate, "left": left, "right": right, "shots": per_rate, **counts}
            )
    work_by_rate = []
    paired = np.zeros((2, per_rate, 11), dtype=float)
    any_unmetered = False
    invalid_work_contributor = False
    for rate_offset, rate in enumerate(_RATES):
        rate_rows = strata[rate]
        policy_values: dict[str, list[int | None]] = {policy_id: [] for policy_id in POLICY_IDS}
        reference_values: list[int | None] = []
        for index, row in enumerate(rate_rows):
            policies_for_row = _policy_map(row)
            action_map = {action["invocation_id"]: action for action in row["actions"]}
            invalid_work_contributor |= any(
                not action_map[identity]["valid"]
                for identity in (
                    "margin_columns_chi8/columns_tol01",
                    "margin_columns_chi8/rows_tol01",
                    "fixed_columns_chi8/columns_chi8",
                )
                if identity in action_map
            )
            if not policies_for_row["margin_columns_chi8"]["gate_accepted"]:
                invalid_work_contributor |= not action_map[
                    "margin_columns_chi8/columns_chi8"
                ]["valid"]
            for policy_id in POLICY_IDS:
                policy_values[policy_id].append(policies_for_row[policy_id]["estimated_arithmetic_flops"])
            reference_values.append(row["reference"]["estimated_arithmetic_flops"])
            margin_work = policy_values["margin_columns_chi8"][-1]
            column_work = policy_values["fixed_columns_chi8"][-1]
            rows_work = policy_values["fixed_rows_tol003"][-1]
            paired[rate_offset, index, :3] = [
                math.nan if margin_work is None else margin_work,
                math.nan if column_work is None else column_work,
                math.nan if rows_work is None else rows_work,
            ]
            paired[rate_offset, index, 3:] = [
                int(policies_for_row["fixed_rows_tol003"]["class_mismatch"]),
                int(policies_for_row["fixed_rows_tol003"]["outcome_discordance"]),
                int(policies_for_row["margin_columns_chi8"]["class_mismatch"]),
                int(policies_for_row["margin_columns_chi8"]["outcome_discordance"]),
                int(policies_for_row["fixed_columns_chi8"]["class_mismatch"]),
                int(policies_for_row["fixed_columns_chi8"]["outcome_discordance"]),
                int(not row["reference"]["certified"]),
                int(bool(row["reference"]["physical_failure"])) if row["reference"]["certified"] else 0,
            ]
        totals = {policy_id: _sum_nullable(values) for policy_id, values in policy_values.items()}
        means = {
            policy_id: (None if total is None else total / per_rate)
            for policy_id, total in totals.items()
        }
        reference_total = _sum_nullable(reference_values)
        study_total = _sum_nullable([*totals.values(), reference_total])
        work_by_rate.append(
            {
                "error_rate": rate,
                "shots": per_rate,
                "policy_totals": totals,
                "policy_means": means,
                "reference_total": reference_total,
                "study_total": study_total,
                "total_work_ratios": {
                    "fixed_rows_tol003_over_fixed_columns_chi8": _work_ratio(totals["fixed_rows_tol003"], totals["fixed_columns_chi8"]),
                    "margin_columns_chi8_over_fixed_columns_chi8": _work_ratio(totals["margin_columns_chi8"], totals["fixed_columns_chi8"]),
                },
            }
        )
        any_unmetered |= any(total is None for total in (*totals.values(), reference_total))
    equal_policy_means = {
        policy_id: (
            None
            if any(item["policy_means"][policy_id] is None for item in work_by_rate)
            else sum(item["policy_means"][policy_id] for item in work_by_rate) / 2
        )
        for policy_id in POLICY_IDS
    }
    reference_mean = (
        None if any(item["reference_total"] is None for item in work_by_rate)
        else sum(item["reference_total"] / per_rate for item in work_by_rate) / 2
    )
    study_mean = (
        None if any(item["study_total"] is None for item in work_by_rate)
        else sum(item["study_total"] / per_rate for item in work_by_rate) / 2
    )
    equal_rate = {
        "policy_means": equal_policy_means,
        "reference_mean": reference_mean,
        "study_mean": study_mean,
        "total_work_ratios": {
            "fixed_rows_tol003_over_fixed_columns_chi8": _work_ratio(equal_policy_means["fixed_rows_tol003"], equal_policy_means["fixed_columns_chi8"]),
            "margin_columns_chi8_over_fixed_columns_chi8": _work_ratio(equal_policy_means["margin_columns_chi8"], equal_policy_means["fixed_columns_chi8"]),
        },
    }
    bootstrap = bootstrap_saving(paired)
    complete = all(item["complete"] for item in sample_counts)
    references_certified = all(item["reference_uncertified"] == 0 for item in invalid_counts)
    execution_eligible = not fixture and complete and references_certified and not any_unmetered
    policy_gate_status: dict[str, str] = {}
    policy_passes: dict[str, bool] = {}
    for policy_id in _SELECTED_POLICIES:
        gates_pass = all(row["gate_passed"] for row in primary if row["policy_id"] == policy_id)
        policy_passes[policy_id] = gates_pass
        if fixture:
            status = "fixture_only"
        elif not execution_eligible:
            status = "blocked_execution"
        elif not gates_pass:
            status = "failed_discrepancy_budget"
        else:
            status = "gates_passed_pending_replay"
        policy_gate_status[policy_id] = status
    numerical_work_pass = bool(bootstrap["passed"])
    margin_work_pass = (
        execution_eligible
        and not invalid_work_contributor
        and policy_passes["margin_columns_chi8"]
        and numerical_work_pass
    )
    bootstrap["passed"] = margin_work_pass
    reasons = []
    if fixture:
        reasons.append("fixture_only")
    if not references_certified:
        reasons.append("uncertified_reference")
    if any_unmetered:
        reasons.append("unmetered_work")
    if invalid_work_contributor:
        reasons.append("invalid_margin_or_comparator_action")
    for policy_id in _SELECTED_POLICIES:
        if not policy_passes[policy_id]:
            reasons.append(f"{policy_id}_discrepancy_budget_failed")
    if bootstrap["status"] == "available" and not numerical_work_pass:
        reasons.append("margin_work_saving_threshold_failed")
    return {
        "sample_counts": sample_counts,
        "invalid_counts": invalid_counts,
        "primary": primary,
        "policies": policies,
        "comparator": comparator,
        "coverage": coverage,
        "paired_physical": paired_physical,
        "work": {"by_rate": work_by_rate, "equal_rate": equal_rate, "bootstrap": bootstrap},
        "decision": {
            "execution_eligible": execution_eligible,
            "policy_gate_status": policy_gate_status,
            "both_policy_gates_passed": all(policy_passes.values()),
            "margin_work_gate_passed": margin_work_pass,
            "reasons": reasons,
        },
    }
