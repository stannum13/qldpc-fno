from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qldpc_fno.decision.study import run_exact_decision_study

CONFIG_PATH = Path("configs/prediction_to_decision.json")


def test_study_exhausts_channels_syndromes_and_actions(tmp_path: Path) -> None:
    payload = run_exact_decision_study(CONFIG_PATH, tmp_path)

    assert payload["schema_version"] == 1
    assert payload["exact_enumeration"] is True
    assert len(payload["action_table"]) == 2 * 8 * 6
    assert {row["action_id"] for row in payload["action_table"]} == {
        "nominal_uniform/physical_map",
        "nominal_uniform/coset_map",
        "correct_global/physical_map",
        "correct_global/coset_map",
        "correct_spatial/physical_map",
        "correct_spatial/coset_map",
    }

    written = json.loads((tmp_path / "action_table.json").read_text())
    assert written == payload
    assert len(payload["provenance"]["config_sha256"]) == 64
    assert len(payload["code"]["identity_sha256"]) == 64


def test_counterexample_gives_coset_aggregation_oracle_headroom(tmp_path: Path) -> None:
    payload = run_exact_decision_study(CONFIG_PATH, tmp_path)
    rows = [
        row
        for row in payload["action_table"]
        if row["true_channel"] == "heterogeneous_counterexample"
        and row["syndrome"] == [0, 1, 1]
        and row["prior_id"] == "correct_spatial"
    ]
    by_decoder = {row["decoder"]: row for row in rows}

    assert by_decoder["physical_map"]["selected_logical_class"] == 0
    assert by_decoder["coset_map"]["selected_logical_class"] == 1
    assert (
        by_decoder["coset_map"]["conditional_bayes_risk"]
        < by_decoder["physical_map"]["conditional_bayes_risk"]
    )
    assert payload["gates"]["open_probability_mass_inference"] is True
    assert payload["gates"]["open_learned_action_selection"] is False
    assert payload["gates"]["observable_policy_oracle_absolute_improvement"] == 0.0
    assert (
        "correct_spatial/coset_map" in payload["gates"]["observable_policy_common_optimal_actions"]
    )
    assert payload["gates"]["observable_policy_requires_state_dependent_action"] is False


def test_true_syndrome_probabilities_normalize_once_per_channel(tmp_path: Path) -> None:
    payload = run_exact_decision_study(CONFIG_PATH, tmp_path)

    for channel in ("uniform", "heterogeneous_counterexample"):
        rows = [
            row
            for row in payload["action_table"]
            if row["true_channel"] == channel and row["action_id"] == "nominal_uniform/physical_map"
        ]
        assert np.isclose(sum(row["syndrome_probability"] for row in rows), 1.0)


def test_artifact_is_byte_identical_across_output_directories(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    run_exact_decision_study(CONFIG_PATH, first)
    run_exact_decision_study(CONFIG_PATH, second)

    assert (first / "action_table.json").read_bytes() == (second / "action_table.json").read_bytes()


def test_artifact_declares_cost_model_risk_counts_and_source_identity(tmp_path: Path) -> None:
    payload = run_exact_decision_study(CONFIG_PATH, tmp_path)
    rows = payload["action_table"]

    assert len(payload["provenance"]["source_sha256"]) == 2
    assert all(len(digest) == 64 for digest in payload["provenance"]["source_sha256"].values())
    assert "not a latency" in payload["operation_count_model"]["claim_boundary"]
    assert {row["operation_count"] for row in rows if row["syndrome"] == [0, 0, 0]} == {176}
    for summary in payload["summaries"]:
        for effect in summary["prior_effects_against_nominal"]:
            assert "conditional_bayes_risk_change_count" in effect
