from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from qecsim.models.planar import PlanarCode

from qldpc_fno.decision import planar_shot_data
from qldpc_fno.decision import simple_confirmation as confirmation
from qldpc_fno.decision.planar_shot_accuracy import logical_class_recovery, score_recovery
from qldpc_fno.decision.simple_confirmation import (
    FIXTURE_SHOT_CONFIG,
    FROZEN_CONFIG,
    FROZEN_SHOT_CONFIG,
    SCIENTIFIC_DOMAIN,
    bootstrap_saving,
    event_indicators,
    load_config,
    load_shot_config,
    primary_upper,
    summarize,
    validate_result_rows,
)

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "configs" / "simple_planar_confirmation.json"
_SHOTS = _ROOT / "configs" / "simple_planar_confirmation_shots.json"
_FIXTURE_SHOTS = _ROOT / "configs" / "simple_planar_confirmation_fixture_shots.json"


def _summary_work(value: int) -> dict[str, object]:
    return {
        "pairwise_contractions": 0,
        "truncation_calls": 0,
        "svd_calls": 0,
        "qr_calls": 0,
        "einsum_calls": 1,
        "einsum_estimated_flops": value,
        "pairwise_output_elements": 0,
        "decomposition_input_elements": 0,
        "estimated_dense_decomposition_flops": 0,
        "estimated_arithmetic_flops": value,
        "terminal_residual_estimated_arithmetic_flops": value,
        "decomposition_attempts": [],
        "peak_observed_array_elements": 0,
        "truncation_events": [],
        "contraction_sweeps": [],
    }


def _summary_action(invocation_id: str, action_id: str, work: int) -> dict[str, object]:
    specs = {
        "rows_tol003": ("rows", None, 0.003),
        "columns_tol01": ("columns", None, 0.01),
        "rows_tol01": ("rows", None, 0.01),
        "columns_chi8": ("columns", 8, None),
        "exact_columns": ("columns", None, None),
        "exact_rows": ("rows", None, None),
    }
    mode, chi, tol = specs[action_id]
    return {
        "invocation_id": invocation_id,
        "action_id": action_id,
        "mode": mode,
        "chi": chi,
        "tol": tol,
        "valid": True,
        "exception_type": None,
        "exception_message": None,
        "masses": [0.8, 0.1, 0.06, 0.04],
        "probabilities": [0.8, 0.1, 0.06, 0.04],
        "selected_class": 0,
        "work": _summary_work(work),
        "estimated_arithmetic_flops": work,
    }


def _summary_row(
    rate: float,
    index: int,
    *,
    rows_mismatch: bool = False,
    margin_mismatch: bool = False,
    reference_certified: bool = True,
    margin_accepted: bool = True,
) -> dict[str, object]:
    code = PlanarCode(5, 5)
    zero_error = np.zeros(82, dtype=np.uint8)
    zero_syndrome = np.zeros(40, dtype=np.uint8)
    order = confirmation.arm_order(index)
    policies: list[dict[str, object]] = []
    actions_by_policy: dict[str, list[dict[str, object]]] = {}
    for policy_id in order:
        if policy_id == "fixed_rows_tol003":
            action_ids = ["rows_tol003"]
            selected_class = 1 if rows_mismatch else 0
            gate = None
            costs = [10]
        elif policy_id == "margin_columns_chi8":
            action_ids = ["columns_tol01", "rows_tol01"]
            if not margin_accepted:
                action_ids.append("columns_chi8")
            selected_class = 1 if margin_mismatch else 0
            gate = margin_accepted
            costs = [10, 10, 100][: len(action_ids)]
        else:
            action_ids = ["columns_chi8"]
            selected_class = 0
            gate = None
            costs = [100]
        policy_actions = [
            _summary_action(f"{policy_id}/{action_id}", action_id, cost)
            for action_id, cost in zip(action_ids, costs, strict=True)
        ]
        selected_action = policy_actions[0] if gate is not False else policy_actions[-1]
        if selected_class == 1:
            selected_action["masses"] = [0.1, 0.8, 0.06, 0.04]
            selected_action["probabilities"] = [0.1, 0.8, 0.06, 0.04]
            selected_action["selected_class"] = 1
            if gate is True:
                for probe in policy_actions[:2]:
                    probe["masses"] = [0.1, 0.8, 0.06, 0.04]
                    probe["probabilities"] = [0.1, 0.8, 0.06, 0.04]
                    probe["selected_class"] = 1
        if gate is False:
            policy_actions[1]["masses"] = [0.1, 0.8, 0.06, 0.04]
            policy_actions[1]["probabilities"] = [0.1, 0.8, 0.06, 0.04]
            policy_actions[1]["selected_class"] = 1
        actions_by_policy[policy_id] = policy_actions
        mismatch = not reference_certified or selected_class != 0
        recovery = logical_class_recovery(code, zero_syndrome, selected_class)
        score = score_recovery(code, zero_error, zero_syndrome, recovery)
        policies.append(
            {
                "policy_id": policy_id,
                "action_invocation_ids": [a["invocation_id"] for a in policy_actions],
                "gate_accepted": gate,
                "selected_class": selected_class,
                "recovery_bsf": recovery.tolist(),
                "syndrome_valid": True,
                "logical_signature": score["logical_signature"].tolist(),
                "physical_failure": bool(score["logical_failure"]),
                "valid_recovery": True,
                "class_mismatch": mismatch,
                "outcome_discordance": mismatch,
                "estimated_arithmetic_flops": sum(costs),
            }
        )
    reference_order = ["columns", "rows"] if index % 2 == 0 else ["rows", "columns"]
    reference_actions = [
        _summary_action(f"reference/exact_{mode}", f"exact_{mode}", 50)
        for mode in reference_order
    ]
    if not reference_certified:
        reference_actions[-1]["masses"] = [0.1, 0.8, 0.06, 0.04]
        reference_actions[-1]["probabilities"] = [0.1, 0.8, 0.06, 0.04]
        reference_actions[-1]["selected_class"] = 1
    return {
        "shot_id": f"d5/p{rate:.6f}/i{index:06d}",
        "error_rate": rate,
        "shot_index": index,
        "sampler_seed": 100_000 + int(rate * 100) * 10_000 + index,
        "error_bsf": [0] * 82,
        "syndrome": [0] * 40,
        "arm_order": list(order),
        "reference_order": reference_order,
        "actions": [
            action
            for policy_id in order
            for action in actions_by_policy[policy_id]
        ]
        + reference_actions,
        "policies": policies,
        "reference": {
            "action_invocation_ids": [a["invocation_id"] for a in reference_actions],
            "certified": reference_certified,
            "certificate": (
                {
                    "selected_class": 0,
                    "maximum_probability_discrepancy": 0.0,
                    "maximum_log_ratio_discrepancy": 0.0,
                    "probability_margins": [0.7, 0.7],
                }
                if reference_certified
                else None
            ),
            "exception_type": None if reference_certified else "ValueError",
            "exception_message": None if reference_certified else "uncertified",
            "probabilities": [0.8, 0.1, 0.06, 0.04] if reference_certified else None,
            "selected_class": 0 if reference_certified else None,
            "recovery_bsf": [0] * 82 if reference_certified else None,
            "syndrome_valid": reference_certified,
            "logical_signature": [0, 0] if reference_certified else None,
            "physical_failure": False if reference_certified else None,
            "estimated_arithmetic_flops": 100,
        },
    }


def _summary_rows(*, fixture: bool = False) -> list[dict[str, object]]:
    count = 2 if fixture else 2048
    return [_summary_row(rate, index) for rate in (0.1, 0.15) for index in range(count)]


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload))
    return path


def test_config_freezes_every_confirmation_constant() -> None:
    config = load_config(_CONFIG)

    assert config == FROZEN_CONFIG
    assert config["contract_id"] == "simple_planar_confirmation_v1"
    assert config["policy_ids"] == ["fixed_rows_tol003", "margin_columns_chi8"]
    assert config["comparator_id"] == "fixed_columns_chi8"
    assert config["primary_endpoints"] == ["class_mismatch", "outcome_discordance"]
    assert config["primary_alpha"] == 0.00625
    assert config["discrepancy_budget"] == 0.005
    assert config["bootstrap_seed"] == 2713269656809941213
    assert config["bootstrap_replicates"] == 10_000
    assert config["minimum_work_saving"] == 0.5
    assert [item["action_id"] for item in config["actions"]] == [
        "rows_tol003",
        "columns_tol01",
        "rows_tol01",
        "columns_chi8",
        "exact_columns",
        "exact_rows",
    ]


def test_shot_configs_are_separate_frozen_identities() -> None:
    scientific = load_shot_config(_SHOTS, fixture=False)
    fixture = load_shot_config(_FIXTURE_SHOTS, fixture=True)

    assert scientific == FROZEN_SHOT_CONFIG
    assert fixture == FIXTURE_SHOT_CONFIG
    assert scientific["shots_per_rate"] == 2048
    assert fixture["shots_per_rate"] == 2
    assert scientific["seed_domain"] != fixture["seed_domain"]


def test_mutating_exported_analysis_mapping_cannot_redefine_validator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = json.loads(_CONFIG.read_text())
    payload["margin_threshold"] = 0.25
    monkeypatch.setitem(confirmation.FROZEN_CONFIG, "margin_threshold", 0.25)

    with pytest.raises(ValueError):
        load_config(_write_json(tmp_path / "config.json", payload))


def test_mutating_exported_nested_lists_cannot_redefine_validator(
    tmp_path: Path,
) -> None:
    payload = json.loads(_CONFIG.read_text())
    payload["required_error_rates"][0] = 0.05
    exported_rates = confirmation.FROZEN_CONFIG["required_error_rates"]
    exported_rates[0] = 0.05
    try:
        with pytest.raises(ValueError):
            load_config(_write_json(tmp_path / "config.json", payload))
    finally:
        exported_rates[0] = 0.1


def test_mutating_exported_nested_action_cannot_redefine_validator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = json.loads(_CONFIG.read_text())
    payload["actions"][0]["tol"] = 0.004
    monkeypatch.setitem(confirmation.FROZEN_CONFIG["actions"][0], "tol", 0.004)

    with pytest.raises(ValueError):
        load_config(_write_json(tmp_path / "config.json", payload))


def test_mutating_exported_shot_mapping_cannot_redefine_validator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = json.loads(_SHOTS.read_text())
    payload["shots_per_rate"] = 1024
    monkeypatch.setitem(confirmation.FROZEN_SHOT_CONFIG, "shots_per_rate", 1024)

    with pytest.raises(ValueError):
        load_shot_config(_write_json(tmp_path / "shots.json", payload), fixture=False)


def test_mutating_exported_shot_nested_list_cannot_redefine_validator(
    tmp_path: Path,
) -> None:
    payload = json.loads(_FIXTURE_SHOTS.read_text())
    payload["error_rates"][1] = 0.2
    exported_rates = confirmation.FIXTURE_SHOT_CONFIG["error_rates"]
    exported_rates[1] = 0.2
    try:
        with pytest.raises(ValueError):
            load_shot_config(_write_json(tmp_path / "shots.json", payload), fixture=True)
    finally:
        exported_rates[1] = 0.15


def test_loaded_config_mutation_does_not_affect_later_load() -> None:
    first = load_config(_CONFIG)
    first["actions"][0]["tol"] = 0.004
    first["required_error_rates"][0] = 0.05

    second = load_config(_CONFIG)
    assert second["actions"][0]["tol"] == 0.003
    assert second["required_error_rates"] == [0.1, 0.15]


@pytest.mark.parametrize(
    ("export_name", "config_key", "changed"),
    [
        ("SCIENTIFIC_DOMAIN", "required_shot_seed_domain", "changed/scientific"),
        ("FIXTURE_DOMAIN", "fixture_seed_domain", "changed/fixture"),
        ("BOOTSTRAP_DOMAIN", "bootstrap_domain", "changed/bootstrap"),
        ("BOOTSTRAP_SEED", "bootstrap_seed", 1),
        ("MARGIN_THRESHOLD", "margin_threshold", 0.25),
    ],
)
def test_rebinding_exported_analysis_scalar_cannot_redefine_validator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    export_name: str,
    config_key: str,
    changed: object,
) -> None:
    payload = json.loads(_CONFIG.read_text())
    payload[config_key] = changed
    monkeypatch.setattr(confirmation, export_name, changed)

    with pytest.raises(ValueError):
        load_config(_write_json(tmp_path / "config.json", payload))


@pytest.mark.parametrize(
    ("fixture", "export_name", "config_key", "changed"),
    [
        (False, "SCIENTIFIC_DOMAIN", "seed_domain", "changed/scientific"),
        (False, "SCIENTIFIC_CAMPAIGN_SEED", "campaign_seed", 1),
        (True, "FIXTURE_DOMAIN", "seed_domain", "changed/fixture"),
        (True, "FIXTURE_CAMPAIGN_SEED", "campaign_seed", 1),
    ],
)
def test_rebinding_exported_shot_scalar_cannot_redefine_validator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fixture: bool,
    export_name: str,
    config_key: str,
    changed: object,
) -> None:
    source = _FIXTURE_SHOTS if fixture else _SHOTS
    payload = json.loads(source.read_text())
    payload[config_key] = changed
    monkeypatch.setattr(confirmation, export_name, changed)

    with pytest.raises(ValueError):
        load_shot_config(_write_json(tmp_path / "shots.json", payload), fixture=fixture)


@pytest.mark.parametrize("count", [2, 2047, 2049, True, 2048.0])
def test_scientific_count_is_frozen(tmp_path: Path, count: object) -> None:
    payload = dict(FROZEN_SHOT_CONFIG, shots_per_rate=count)
    path = _write_json(tmp_path / "shots.json", payload)
    with pytest.raises(ValueError):
        load_shot_config(path, fixture=False)


@pytest.mark.parametrize("count", [1, 3, 2048, True, 2.0])
def test_fixture_count_is_frozen(tmp_path: Path, count: object) -> None:
    payload = dict(FIXTURE_SHOT_CONFIG, shots_per_rate=count)
    path = _write_json(tmp_path / "shots.json", payload)
    with pytest.raises(ValueError):
        load_shot_config(path, fixture=True)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("contract_id", "changed"),
        ("required_shot_seed_domain", "changed"),
        ("fixture_seed_domain", "changed"),
        ("code_distance", 5.0),
        ("noise_model", "changed"),
        ("required_error_rates", [0.15, 0.1]),
        ("required_shots_per_rate", 2048.0),
        ("policy_ids", ["margin_columns_chi8", "fixed_rows_tol003"]),
        ("comparator_id", "fixed_rows_tol003"),
        ("margin_threshold", 0.3),
        ("reference_probability_tolerance", 1e-9),
        ("reference_log_ratio_tolerance", 1e-7),
        ("reference_margin_discrepancy_factor", 1.0),
        ("primary_endpoints", ["outcome_discordance", "class_mismatch"]),
        ("family_alpha", 0.1),
        ("primary_alpha", 0.0125),
        ("discrepancy_budget", 0.01),
        ("bootstrap_domain", "changed"),
        ("bootstrap_seed", 0),
        ("bootstrap_replicates", 9999),
        ("bootstrap_generator", "MT19937"),
        ("bootstrap_quantile", 0.1),
        ("bootstrap_quantile_method", "nearest"),
        ("minimum_work_saving", 0.49),
        ("order_rule", "changed"),
        ("invalidity_rule", "changed"),
        ("status_rule", "changed"),
    ],
)
def test_config_rejects_every_changed_constant(tmp_path: Path, key: str, value: object) -> None:
    payload = deepcopy(FROZEN_CONFIG)
    payload[key] = value
    with pytest.raises(ValueError):
        load_config(_write_json(tmp_path / "config.json", payload))


@pytest.mark.parametrize("key", list(FROZEN_CONFIG))
def test_config_rejects_every_missing_key(tmp_path: Path, key: str) -> None:
    payload = deepcopy(FROZEN_CONFIG)
    del payload[key]
    with pytest.raises(ValueError):
        load_config(_write_json(tmp_path / "config.json", payload))


def test_config_rejects_unknown_and_duplicate_keys(tmp_path: Path) -> None:
    payload = deepcopy(FROZEN_CONFIG)
    payload["unexpected"] = True
    with pytest.raises(ValueError):
        load_config(_write_json(tmp_path / "unknown.json", payload))

    duplicate = _CONFIG.read_text().replace(
        '"schema_version": 1,', '"schema_version": 1,\n  "schema_version": 1,', 1
    )
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        load_config(path)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_config_rejects_nonfinite_json_constants(tmp_path: Path, literal: str) -> None:
    text = _CONFIG.read_text().replace("0.30710401263493464", literal)
    path = tmp_path / "nonfinite.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize(
    "actions",
    [
        [],
        FROZEN_CONFIG["actions"][:-1],
        list(reversed(FROZEN_CONFIG["actions"])),
        [
            *FROZEN_CONFIG["actions"][:-1],
            {"action_id": "exact_rows", "mode": "rows", "chi": 8, "tol": None},
        ],
    ],
)
def test_config_rejects_changed_or_insufficient_action_inventory(
    tmp_path: Path, actions: object
) -> None:
    payload = deepcopy(FROZEN_CONFIG)
    payload["actions"] = actions
    with pytest.raises(ValueError):
        load_config(_write_json(tmp_path / "config.json", payload))


def test_action_rejects_unknown_missing_and_duplicate_fields(tmp_path: Path) -> None:
    for mutation in ("unknown", "missing"):
        payload = deepcopy(FROZEN_CONFIG)
        if mutation == "unknown":
            payload["actions"][0]["unexpected"] = True
        else:
            del payload["actions"][0]["mode"]
        with pytest.raises(ValueError):
            load_config(_write_json(tmp_path / f"{mutation}.json", payload))

    text = _CONFIG.read_text().replace(
        '"action_id": "rows_tol003",',
        '"action_id": "rows_tol003", "action_id": "rows_tol003",',
        1,
    )
    path = tmp_path / "duplicate-action.json"
    path.write_text(text)
    with pytest.raises(ValueError, match="duplicate"):
        load_config(path)


@pytest.mark.parametrize("fixture", [True, False])
def test_shot_config_rejects_wrong_mode_domain(tmp_path: Path, fixture: bool) -> None:
    payload = FROZEN_SHOT_CONFIG if fixture else FIXTURE_SHOT_CONFIG
    with pytest.raises(ValueError):
        load_shot_config(_write_json(tmp_path / "shots.json", payload), fixture=fixture)


@pytest.mark.parametrize("fixture", [None, 0, 1, "yes"])
def test_shot_config_requires_boolean_fixture_flag(tmp_path: Path, fixture: object) -> None:
    path = _write_json(tmp_path / "shots.json", FROZEN_SHOT_CONFIG)
    with pytest.raises(TypeError):
        load_shot_config(path, fixture=fixture)  # type: ignore[arg-type]


def test_shot_config_rejects_unknown_missing_duplicate_and_nonfinite(tmp_path: Path) -> None:
    payload = dict(FROZEN_SHOT_CONFIG, unexpected=True)
    with pytest.raises(ValueError):
        load_shot_config(_write_json(tmp_path / "unknown.json", payload), fixture=False)

    payload = dict(FROZEN_SHOT_CONFIG)
    del payload["noise_model"]
    with pytest.raises(ValueError):
        load_shot_config(_write_json(tmp_path / "missing.json", payload), fixture=False)

    duplicate = _SHOTS.read_text().replace(
        '"schema_version": 1,', '"schema_version": 1,\n  "schema_version": 1,', 1
    )
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        load_shot_config(path, fixture=False)

    path = tmp_path / "nonfinite.json"
    path.write_text(_SHOTS.read_text().replace("0.1", "NaN", 1))
    with pytest.raises(ValueError):
        load_shot_config(path, fixture=False)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema_version", True),
        ("campaign_seed", True),
        ("campaign_seed", 0),
        ("code_distance", 5.0),
        ("error_rates", [0.15, 0.1]),
        ("noise_model", "changed"),
    ],
)
def test_shot_config_rejects_changed_identity(
    tmp_path: Path, key: str, value: object
) -> None:
    payload = dict(FROZEN_SHOT_CONFIG)
    payload[key] = value
    with pytest.raises(ValueError):
        load_shot_config(_write_json(tmp_path / "shots.json", payload), fixture=False)


def test_summary_builds_exact_eight_primary_rows_and_rejects_three_events() -> None:
    rows = _summary_rows()
    for index in range(3):
        rows[index] = _summary_row(0.1, index, rows_mismatch=True)

    summary = summarize(rows, fixture=False)

    assert len(summary["primary"]) == 8
    target = summary["primary"][0]
    assert target["events"] == 3 and target["shots"] == 2048
    assert target["upper_bound"] == pytest.approx(0.005205093018026918)
    assert target["gate_passed"] is False
    assert summary["decision"]["policy_gate_status"]["fixed_rows_tol003"] == (
        "failed_discrepancy_budget"
    )
    assert summary["decision"]["policy_gate_status"]["margin_columns_chi8"] == (
        "gates_passed_pending_replay"
    )


@pytest.mark.parametrize("events", [0, 1, 2, 3])
def test_summary_primary_event_budget_matches_exact_cp_boundary(events: int) -> None:
    rows = _summary_rows()
    for index in range(events):
        rows[index] = _summary_row(0.1, index, margin_mismatch=True)

    target = summarize(rows, fixture=False)["primary"][4]

    assert target["events"] == events
    assert target["gate_passed"] is (events <= 2)


def test_summary_has_frozen_nested_schema_and_equal_rate_work_estimands() -> None:
    rows = _summary_rows()
    for index in range(100):
        rows[index] = _summary_row(0.1, index, margin_accepted=False)
    for index in range(400):
        offset = 2048 + index
        rows[offset] = _summary_row(0.15, index, margin_accepted=False)

    summary = summarize(rows, fixture=False)

    assert set(summary) == {
        "sample_counts", "invalid_counts", "primary", "policies", "comparator",
        "coverage", "paired_physical", "work", "decision",
    }
    assert set(summary["sample_counts"][0]) == {
        "error_rate", "expected", "observed", "complete",
    }
    assert set(summary["invalid_counts"][0]) == {
        "error_rate", "reference_uncertified", "invalid_recoveries", "invalid_actions",
        "unmetered_actions",
    }
    assert set(summary["primary"][0]) == {
        "policy_id", "error_rate", "endpoint", "events", "shots", "alpha",
        "upper_bound", "threshold", "gate_passed",
    }
    assert set(summary["policies"][0]) == {
        "policy_id", "error_rate", "shots", "valid_recoveries", "physical_failures",
        "physical_failure_rate",
    }
    assert set(summary["comparator"][0]) == {
        "error_rate", "shots", "class_mismatches", "outcome_discordances",
        "class_mismatch_rate", "outcome_discordance_rate",
    }
    assert set(summary["coverage"][0]) == {
        "error_rate", "shots", "accepted", "escalated", "accepted_class_mismatches",
        "accepted_outcome_discordances", "coverage", "accepted_class_mismatch_rate",
        "accepted_outcome_discordance_rate",
    }
    assert set(summary["paired_physical"][0]) == {
        "error_rate", "left", "right", "shots", "both_succeed", "left_only_fails",
        "right_only_fails", "both_fail", "uncertified_reference",
    }
    assert set(summary["work"]) == {"by_rate", "equal_rate", "bootstrap"}
    assert set(summary["work"]["by_rate"][0]) == {
        "error_rate", "shots", "policy_totals", "policy_means", "reference_total",
        "study_total", "total_work_ratios",
    }
    assert set(summary["work"]["equal_rate"]) == {
        "policy_means", "reference_mean", "study_mean", "total_work_ratios",
    }
    assert set(summary["work"]["bootstrap"]) == {
        "status", "estimand", "replicates", "seed", "bit_generator", "quantile_method",
        "estimate", "lower_bound", "threshold", "passed", "unavailable_reason",
    }
    assert set(summary["decision"]) == {
        "execution_eligible", "policy_gate_status", "both_policy_gates_passed",
        "margin_work_gate_passed", "reasons",
    }
    assert [row["policy_id"] for row in summary["primary"]] == [
        "fixed_rows_tol003", "fixed_rows_tol003", "fixed_rows_tol003", "fixed_rows_tol003",
        "margin_columns_chi8", "margin_columns_chi8", "margin_columns_chi8",
        "margin_columns_chi8",
    ]
    assert [row["accepted"] for row in summary["coverage"]] == [1948, 1648]
    by_rate = summary["work"]["by_rate"]
    assert by_rate[0]["policy_means"]["margin_columns_chi8"] == pytest.approx(
        (1948 * 20 + 100 * 120) / 2048
    )
    assert by_rate[1]["policy_means"]["margin_columns_chi8"] == pytest.approx(
        (1648 * 20 + 400 * 120) / 2048
    )
    assert summary["work"]["equal_rate"]["policy_means"]["margin_columns_chi8"] == (
        pytest.approx(
            (
                by_rate[0]["policy_means"]["margin_columns_chi8"]
                + by_rate[1]["policy_means"]["margin_columns_chi8"]
            )
            / 2
        )
    )
    assert summary["work"]["bootstrap"]["estimate"] == pytest.approx(
        ((1948 * 0.8 + 100 * -0.2) / 2048 + (1648 * 0.8 + 400 * -0.2) / 2048) / 2
    )


def test_fixture_summary_is_noninferential_and_keeps_numerical_rows_visible() -> None:
    summary = summarize(_summary_rows(fixture=True), fixture=True)

    assert len(summary["primary"]) == 8
    assert all(row["shots"] == 2 for row in summary["primary"])
    assert summary["decision"]["execution_eligible"] is False
    assert summary["decision"]["policy_gate_status"] == {
        "fixed_rows_tol003": "fixture_only",
        "margin_columns_chi8": "fixture_only",
    }
    assert summary["work"]["bootstrap"]["status"] == "fixture_only"


def test_uncertified_reference_forces_events_blocks_status_and_stays_out_of_pairs() -> None:
    rows = _summary_rows()
    rows[0] = _summary_row(0.1, 0, reference_certified=False)

    summary = summarize(rows, fixture=False)

    assert [row["events"] for row in summary["primary"] if row["error_rate"] == 0.1] == [
        1, 1, 1, 1,
    ]
    assert summary["coverage"][0]["accepted_class_mismatches"] == 1
    assert summary["coverage"][0]["accepted_outcome_discordances"] == 1
    assert summary["decision"]["execution_eligible"] is False
    assert set(summary["decision"]["policy_gate_status"].values()) == {"blocked_execution"}
    reference_pairs = [row for row in summary["paired_physical"] if row["right"] == "reference"]
    assert all(row["uncertified_reference"] == (1 if row["error_rate"] == 0.1 else 0) for row in reference_pairs)
    assert all(
        row["both_succeed"] + row["left_only_fails"] + row["right_only_fails"]
        + row["both_fail"] + row["uncertified_reference"] == 2048
        for row in reference_pairs
    )


def test_zero_coverage_uses_null_conditional_rates() -> None:
    rows = [
        _summary_row(rate, index, margin_accepted=False)
        for rate in (0.1, 0.15)
        for index in range(2048)
    ]

    summary = summarize(rows, fixture=False)

    assert all(row["accepted"] == 0 for row in summary["coverage"])
    assert all(row["accepted_class_mismatch_rate"] is None for row in summary["coverage"])
    assert all(row["accepted_outcome_discordance_rate"] is None for row in summary["coverage"])


def test_invalid_probe_with_valid_fallback_is_counted_as_action_not_recovery() -> None:
    rows = _summary_rows()
    row = _summary_row(0.1, 0, margin_accepted=False)
    probe = next(
        action for action in row["actions"]
        if action["invocation_id"] == "margin_columns_chi8/columns_tol01"
    )
    probe.update(
        valid=False,
        exception_type="InvalidCosetMassError",
        exception_message="invalid",
        masses=None,
        probabilities=None,
        selected_class=None,
    )
    rows[0] = row

    summary = summarize(rows, fixture=False)

    invalid = summary["invalid_counts"][0]
    assert invalid["invalid_actions"]["margin_columns_chi8/columns_tol01"] == 1
    assert invalid["invalid_recoveries"]["margin_columns_chi8"] == 0


@pytest.mark.parametrize(
    "mutation",
    ["missing_shot", "duplicate_index", "missing_action", "missing_trace"],
)
def test_result_validator_rejects_incomplete_or_unmetered_rows(mutation: str) -> None:
    rows = _summary_rows(fixture=True)
    if mutation == "missing_shot":
        rows.pop()
    elif mutation == "duplicate_index":
        rows[-1] = deepcopy(rows[-2])
    elif mutation == "missing_action":
        rows[0]["actions"].pop()
    elif mutation == "missing_trace":
        del rows[0]["actions"][0]["work"]["decomposition_attempts"]
    with pytest.raises((TypeError, ValueError)):
        validate_result_rows(rows, fixture=True)


def test_unknown_work_is_retained_and_blocks_positive_execution_status() -> None:
    rows = _summary_rows()
    action = rows[0]["actions"][0]
    policy = rows[0]["policies"][0]
    action["estimated_arithmetic_flops"] = None
    action["work"] = None
    policy["estimated_arithmetic_flops"] = None

    summary = summarize(rows, fixture=False)

    assert summary["invalid_counts"][0]["unmetered_actions"][action["invocation_id"]] == 1
    assert summary["work"]["by_rate"][0]["policy_totals"][policy["policy_id"]] is None
    assert summary["decision"]["execution_eligible"] is False
    assert set(summary["decision"]["policy_gate_status"].values()) == {"blocked_execution"}


def test_invalid_contributing_probe_suppresses_only_secondary_work_claim() -> None:
    rows = _summary_rows()
    row = _summary_row(0.1, 0, margin_accepted=False)
    probe = next(
        action for action in row["actions"]
        if action["invocation_id"] == "margin_columns_chi8/columns_tol01"
    )
    probe.update(
        valid=False,
        exception_type="InvalidCosetMassError",
        exception_message="invalid",
        masses=None,
        probabilities=None,
        selected_class=None,
    )
    rows[0] = row

    summary = summarize(rows, fixture=False)

    assert summary["decision"]["execution_eligible"] is True
    assert summary["work"]["bootstrap"]["status"] == "available"
    assert summary["work"]["bootstrap"]["passed"] is False
    assert summary["decision"]["margin_work_gate_passed"] is False


def test_validator_recomputes_persisted_events_and_logical_scores() -> None:
    rows = _summary_rows(fixture=True)
    rows[0]["policies"][0]["class_mismatch"] = True
    with pytest.raises(ValueError, match="event"):
        validate_result_rows(rows, fixture=True)

    rows = _summary_rows(fixture=True)
    rows[0]["policies"][0]["logical_signature"] = [1, 0]
    rows[0]["policies"][0]["physical_failure"] = False
    with pytest.raises(ValueError, match="logical|physical"):
        validate_result_rows(rows, fixture=True)


def test_fixture_cannot_derive_production_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_hash(*args, **kwargs):
        pytest.fail("must reject scientific test coordinates before hashing")

    monkeypatch.setattr(planar_shot_data.hashlib, "sha256", forbidden_hash)
    with pytest.raises(ValueError, match="scientific confirmation seed forbidden in test mode"):
        planar_shot_data._shot_seed(SCIENTIFIC_DOMAIN, error_rate=0.1, shot_index=0)


def test_fixture_coordinate_can_reach_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    real_hash = planar_shot_data.hashlib.sha256

    def recording_hash(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_hash(*args, **kwargs)

    monkeypatch.setattr(planar_shot_data.hashlib, "sha256", recording_hash)
    seed = planar_shot_data._shot_seed(
        FIXTURE_SHOT_CONFIG["seed_domain"], error_rate=0.1, shot_index=0
    )
    assert type(seed) is int
    assert calls == 1


def test_direct_generator_rejects_scientific_domain_before_sampler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from qecsim.models.generic import DepolarizingErrorModel

    def forbidden(*args, **kwargs):
        pytest.fail("scientific confirmation generator reached RNG or sampler")

    monkeypatch.setattr(planar_shot_data, "_shot_seed", forbidden)
    monkeypatch.setattr(DepolarizingErrorModel, "generate", forbidden)
    with pytest.raises(ValueError, match="scientific confirmation seed forbidden in test mode"):
        planar_shot_data.generate_planar_shots(_SHOTS, tmp_path / "out")


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        (0, 0.0024750442291897544),
        (1, 0.003498836715834498),
        (2, 0.004385297715096555),
        (3, 0.005205093018026918),
        (2048, 1.0),
    ],
)
def test_primary_upper_matches_independent_binomial_inversion(
    events: int, expected: float
) -> None:
    actual = primary_upper(events, 2048)
    assert actual == pytest.approx(expected, abs=1e-15)
    if events < 2048:
        cdf = sum(
            math.comb(2048, j) * actual**j * (1.0 - actual) ** (2048 - j)
            for j in range(events + 1)
        )
        assert cdf == pytest.approx(0.00625, abs=1e-13)


def test_primary_upper_is_monotone_and_freezes_confirmation_cutoff() -> None:
    bounds = [primary_upper(events, 2048) for events in range(5)]
    assert bounds == sorted(bounds)
    assert [bound <= 0.005 for bound in bounds[:4]] == [True, True, True, False]


@pytest.mark.parametrize(
    ("events", "shots"),
    [
        (True, 2048),
        (0, True),
        (0.0, 2048),
        (0, 2048.0),
        (-1, 2048),
        (2049, 2048),
        (0, 0),
        (0, -1),
    ],
)
def test_primary_upper_rejects_noninteger_or_out_of_range_counts(
    events: object, shots: object
) -> None:
    with pytest.raises(ValueError):
        primary_upper(events, shots)  # type: ignore[arg-type]


def test_two_wrong_classes_can_share_failure_outcome() -> None:
    assert event_indicators(
        reference_valid=True,
        policy_valid=True,
        policy_class=1,
        reference_class=2,
        policy_failure=True,
        reference_failure=True,
    ) == (True, False)


@pytest.mark.parametrize(
    ("policy_class", "reference_class", "policy_failure", "reference_failure", "expected"),
    [
        (0, 0, False, False, (False, False)),
        (1, 2, False, False, (True, False)),
        (3, 3, True, False, (False, True)),
        (0, 3, False, True, (True, True)),
    ],
)
def test_event_indicators_keep_class_and_failure_endpoints_distinct(
    policy_class: int,
    reference_class: int,
    policy_failure: bool,
    reference_failure: bool,
    expected: tuple[bool, bool],
) -> None:
    assert event_indicators(
        reference_valid=True,
        policy_valid=True,
        policy_class=policy_class,
        reference_class=reference_class,
        policy_failure=policy_failure,
        reference_failure=reference_failure,
    ) == expected


@pytest.mark.parametrize(
    ("reference_valid", "policy_valid"),
    [(False, True), (True, False), (False, False)],
)
def test_invalid_reference_or_policy_counts_both_primary_events(
    reference_valid: bool, policy_valid: bool
) -> None:
    assert event_indicators(
        reference_valid=reference_valid,
        policy_valid=policy_valid,
        policy_class=0 if policy_valid else None,
        reference_class=0 if reference_valid else None,
        policy_failure=False if policy_valid else None,
        reference_failure=False if reference_valid else None,
    ) == (True, True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_valid": 1},
        {"policy_valid": 0},
        {"policy_class": True},
        {"policy_class": -1},
        {"policy_class": 4},
        {"reference_class": 1.0},
        {"policy_failure": 0},
        {"reference_failure": 1},
        {"policy_class": None},
        {"reference_failure": None},
    ],
)
def test_event_indicators_reject_invalid_validity_and_labels(kwargs: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "reference_valid": True,
        "policy_valid": True,
        "policy_class": 0,
        "reference_class": 0,
        "policy_failure": False,
        "reference_failure": False,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        event_indicators(**arguments)  # type: ignore[arg-type]


def test_event_indicators_require_null_labels_for_invalid_records() -> None:
    with pytest.raises(ValueError):
        event_indicators(
            reference_valid=False,
            policy_valid=True,
            policy_class=0,
            reference_class=0,
            policy_failure=False,
            reference_failure=False,
        )


def _paired(*, margin: float, column: float = 100.0, rows: float = 40.0) -> np.ndarray:
    paired = np.zeros((2, 2048, 11), dtype=np.float64)
    paired[:, :, 0] = margin
    paired[:, :, 1] = column
    paired[:, :, 2] = rows
    return paired


def test_bootstrap_constant_seventy_five_percent_saving_is_exact() -> None:
    result = bootstrap_saving(_paired(margin=25.0))

    assert set(result) == {
        "status",
        "estimand",
        "replicates",
        "seed",
        "bit_generator",
        "quantile_method",
        "estimate",
        "lower_bound",
        "threshold",
        "passed",
        "unavailable_reason",
    }
    assert result == {
        "status": "available",
        "estimand": "equal_rate_mean_paired_relative_saving",
        "replicates": 10_000,
        "seed": 2713269656809941213,
        "bit_generator": "PCG64",
        "quantile_method": "linear",
        "estimate": 0.75,
        "lower_bound": 0.75,
        "threshold": 0.5,
        "passed": True,
        "unavailable_reason": None,
    }


@pytest.mark.parametrize(
    ("margin", "estimate", "passed"),
    [(50.0, 0.5, False), (125.0, -0.25, False)],
)
def test_bootstrap_gate_is_strict_and_preserves_negative_savings(
    margin: float, estimate: float, passed: bool
) -> None:
    result = bootstrap_saving(_paired(margin=margin))
    assert result["estimate"] == pytest.approx(estimate)
    assert result["lower_bound"] == pytest.approx(estimate)
    assert result["passed"] is passed


def test_bootstrap_equal_weights_rates_and_uses_mean_of_paired_ratios() -> None:
    paired = _paired(margin=25.0)
    paired[0, :, 0] = 0.0
    paired[1, :, 0] = 100.0
    paired[0, :, 1] = np.tile([4.0, 100.0], 1024)
    paired[1, :, 1] = np.tile([4.0, 100.0], 1024)
    paired[0, :, 0] = paired[0, :, 1] * 0.25
    paired[1, :, 0] = paired[1, :, 1] * 0.75

    result = bootstrap_saving(paired)
    ratio_of_totals = 1.0 - float(np.sum(paired[:, :, 0])) / float(np.sum(paired[:, :, 1]))
    assert result["estimate"] == pytest.approx(0.5)
    assert ratio_of_totals == pytest.approx(0.5)
    assert result["passed"] is False


def test_bootstrap_mean_ratios_differs_from_ratio_of_totals_when_costs_covary() -> None:
    paired = _paired(margin=25.0)
    for stratum in range(2):
        paired[stratum, :1024, 0] = 0.0
        paired[stratum, :1024, 1] = 1.0
        paired[stratum, 1024:, 0] = 100.0
        paired[stratum, 1024:, 1] = 100.0

    result = bootstrap_saving(paired)
    ratio_of_totals = 1.0 - float(np.sum(paired[:, :, 0])) / float(np.sum(paired[:, :, 1]))
    assert result["estimate"] == pytest.approx(0.5)
    assert ratio_of_totals == pytest.approx(1.0 / 101.0)


def test_bootstrap_is_deterministic_and_pairing_sensitive() -> None:
    paired = _paired(margin=25.0)
    ramp = np.arange(1, 2049, dtype=np.float64)
    paired[:, :, 1] = ramp * 10.0
    paired[:, :, 0] = ramp * np.tile([1.0, 9.0], 1024)
    first = bootstrap_saving(paired)
    second = bootstrap_saving(paired.copy())
    unpaired = paired.copy()
    unpaired[:, :, 0] = unpaired[:, ::-1, 0]
    changed = bootstrap_saving(unpaired)

    assert first == second
    assert changed["estimate"] != pytest.approx(first["estimate"])


def test_bootstrap_draws_complete_paired_rows_with_frozen_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int, int]] = []
    paired_row_slices: list[tuple[object, object, object]] = []
    original_asarray = confirmation.np.asarray

    class TrackingArray(np.ndarray):
        def __getitem__(self, key):
            if (
                isinstance(key, tuple)
                and len(key) == 3
                and type(key[0]) is int
                and isinstance(key[1], np.ndarray)
            ):
                paired_row_slices.append(key)
            return super().__getitem__(key)

    class GeneratorSpy:
        def __init__(self, bit_generator: object) -> None:
            assert type(bit_generator).__name__ == "PCG64"

        def integers(self, low: int, high: int, *, size: int) -> np.ndarray:
            calls.append((low, high, size))
            return np.arange(size, dtype=np.int64)

    def tracking_asarray(value: object, *, dtype: object) -> TrackingArray:
        return original_asarray(value, dtype=dtype).view(TrackingArray)

    monkeypatch.setattr(confirmation.np.random, "Generator", GeneratorSpy)
    monkeypatch.setattr(confirmation.np, "asarray", tracking_asarray)
    result = bootstrap_saving(_paired(margin=25.0))

    assert result["estimate"] == 0.75
    assert result["lower_bound"] == 0.75
    assert calls == [(0, 2048, 2048)] * 20_000
    assert len(paired_row_slices) == 20_000
    assert all(key[2] == slice(None) for key in paired_row_slices)


@pytest.mark.parametrize(
    ("mutator", "reason"),
    [
        (lambda value: value.__setitem__((0, 0, 0), np.nan), "nonfinite_values"),
        (lambda value: value.__setitem__((0, 0, 0), -1.0), "negative_work"),
        (lambda value: value.__setitem__((0, 0, 0), 1.5), "noninteger_work"),
        (lambda value: value.__setitem__((0, 0, 1), 0.0), "nonpositive_comparator_work"),
    ],
)
def test_bootstrap_invalid_or_missing_costs_suppress_inference(mutator, reason: str) -> None:
    paired = _paired(margin=25.0)
    mutator(paired)
    result = bootstrap_saving(paired)
    assert result["status"] == "unavailable"
    assert result["estimate"] is None
    assert result["lower_bound"] is None
    assert result["passed"] is False
    assert result["unavailable_reason"] == reason


@pytest.mark.parametrize(
    "paired",
    [
        np.zeros((2048, 11)),
        np.zeros((2, 2047, 11)),
        np.zeros((2, 2048, 10)),
        np.zeros((2, 2048, 11), dtype=object),
    ],
)
def test_bootstrap_rejects_wrong_shape_or_nonnumeric_array(paired: np.ndarray) -> None:
    with pytest.raises(ValueError):
        bootstrap_saving(paired)


@pytest.mark.parametrize("shots", [2, 2048])
def test_bootstrap_rejects_complex_arrays_before_imaginary_work_is_discarded(
    shots: int,
) -> None:
    paired = np.zeros((2, shots, 11), dtype=np.complex128)
    paired[:, :, 0] = 25.0 + 999.0j
    paired[:, :, 1] = 100.0

    with pytest.raises(ValueError, match="real numeric"):
        bootstrap_saving(paired)


def test_bootstrap_rejects_non_array_input() -> None:
    with pytest.raises(TypeError):
        bootstrap_saving([])  # type: ignore[arg-type]


def test_bootstrap_fixture_shape_never_runs_inference() -> None:
    paired = np.zeros((2, 2, 11), dtype=np.float64)
    paired[:, :, 0] = 25.0
    paired[:, :, 1] = 100.0
    paired[:, :, 2] = 40.0
    result = bootstrap_saving(paired)
    assert result["status"] == "fixture_only"
    assert result["estimate"] is None
    assert result["lower_bound"] is None
    assert result["passed"] is False
    assert result["unavailable_reason"] == "fixture_analysis_is_noninferential"


@pytest.mark.parametrize(
    ("index", "value", "reason"),
    [
        ((0, 0, 0), np.nan, "nonfinite_values"),
        ((0, 0, 0), -1.0, "negative_work"),
        ((0, 0, 0), 1.5, "noninteger_work"),
        ((0, 0, 1), 0.0, "nonpositive_comparator_work"),
        ((0, 0, 3), 2.0, "nonbinary_outcomes"),
    ],
)
def test_bootstrap_fixture_contents_are_validated_before_fixture_status(
    index: tuple[int, int, int], value: float, reason: str
) -> None:
    paired = np.zeros((2, 2, 11), dtype=np.float64)
    paired[:, :, 0] = 25.0
    paired[:, :, 1] = 100.0
    paired[:, :, 2] = 40.0
    paired[index] = value

    result = bootstrap_saving(paired)
    assert result["status"] == "unavailable"
    assert result["unavailable_reason"] == reason
    assert result["estimate"] is None
    assert result["lower_bound"] is None
    assert result["passed"] is False
