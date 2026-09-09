from __future__ import annotations

import json
from pathlib import Path

from qldpc_fno.decision.gflownet_study import (
    _unique_top_class_correct,
    run_gflownet_study,
)

CONFIG = Path("configs/gflownet_mass.json")
DECISION_CONFIG = Path("configs/prediction_to_decision.json")
BASELINE = Path("evidence/mass-sampling-baselines/mass_sampling.json")


def test_induced_ties_are_not_credited_as_correct() -> None:
    assert _unique_top_class_correct([0.5, 0.5], exact_class=0) is False
    assert _unique_top_class_correct([0.4, 0.6], exact_class=1) is True


def _reduced_config(tmp_path: Path) -> Path:
    payload = json.loads(CONFIG.read_text())
    payload["training_seeds"] = payload["training_seeds"][:1]
    payload["sampling_replicate_seeds"] = payload["sampling_replicate_seeds"][:1]
    payload["hidden_width"] = 8
    payload["training_steps"] = 20
    payload["training_syndrome_indices"] = [0]
    payload["heldout_syndrome_indices"] = [1]
    payload["sample_budgets"] = [8]
    payload["training_fields"] = payload["training_fields"][:1]
    path = tmp_path / "gflownet-reduced.json"
    path.write_text(json.dumps(payload))
    return path


def test_reduced_study_keeps_development_and_confirmation_contexts_distinct(
    tmp_path: Path,
) -> None:
    config = _reduced_config(tmp_path)
    payload = run_gflownet_study(config, DECISION_CONFIG, BASELINE, tmp_path / "out")

    assert payload["run_label"] == "reduced_non_scientific"
    assert payload["gate4_status"] == "unresolved_requires_fresh_paired_baselines"
    assert len(payload["training_records"]) == 1
    assert {row["evaluation_split"] for row in payload["evaluation_rows"]} == {
        "training_fit",
        "heldout_syndrome",
        "heldout_field",
    }
    assert all(row["trajectory_multiplicity"] == 1 for row in payload["evaluation_rows"])
    assert all(len(row["sampled_metrics"]) == 1 for row in payload["evaluation_rows"])
    assert len(payload["provenance"]["baseline_sha256"]) == 64
    assert {summary["budget_axis"] for summary in payload["baseline_summaries"]} == {
        "exact",
        "retained",
        "complete_proposals",
    }
    assert payload["sampled_summaries"]
    assert all(
        summary["training_seed_sampling_replicate_unit_count"] == 1
        for summary in payload["sampled_summaries"]
    )
    operation_counts = payload["evaluation_rows"][0]["sampled_metrics"][0]["operation_counts"]
    assert operation_counts["batched_policy_forward_invocations"] == 4
    assert operation_counts["terminal_action_logit_evaluations"] == 8 * 4


def test_reduced_study_replays_byte_identically(tmp_path: Path) -> None:
    config = _reduced_config(tmp_path)

    run_gflownet_study(config, DECISION_CONFIG, BASELINE, tmp_path / "first")
    run_gflownet_study(config, DECISION_CONFIG, BASELINE, tmp_path / "second")

    assert (tmp_path / "first" / "gflownet_mass.json").read_bytes() == (
        tmp_path / "second" / "gflownet_mass.json"
    ).read_bytes()
