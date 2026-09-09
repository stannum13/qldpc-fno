from __future__ import annotations

import json
from pathlib import Path

from qldpc_fno.decision.sampling_study import run_mass_sampling_study

DECISION_CONFIG = Path("configs/prediction_to_decision.json")
SAMPLING_CONFIG = Path("configs/mass_sampling.json")


def test_sampling_study_has_exact_and_budgeted_rows(tmp_path: Path) -> None:
    payload = run_mass_sampling_study(DECISION_CONFIG, SAMPLING_CONFIG, tmp_path)

    assert len(payload["rows"]) == 2 * 8 * (1 + 2 * 4 + 2 * 4)
    exact_rows = [row for row in payload["rows"] if row["method"] == "exact_enumeration"]
    assert len(exact_rows) == 16
    assert all(row["total_variation"] == 0.0 for row in exact_rows)
    assert all(row["physical_state_coverage"] == 1.0 for row in exact_rows)
    assert all("operation_counts" in row for row in payload["rows"])
    assert all(row["operation_counts"]["setup_syndrome_evaluations"] == 128 for row in exact_rows)
    assert {
        row["sample_budget"] for row in payload["rows"] if row["budget_axis"] == "retained"
    } == {
        32,
        128,
        512,
        2048,
    }
    generated_rows = [row for row in payload["rows"] if row["budget_axis"] == "complete_proposals"]
    assert {row["proposal_budget"] for row in generated_rows} == {512, 2048, 8192, 32768}
    assert all(row["proposal_count"] == row["proposal_budget"] for row in generated_rows)
    for summary in payload["summaries"]:
        assert "mean_effective_sample_size" not in summary
        assert 0 <= summary["ess_available_state_count"] <= summary["state_count"]
        assert "mean_available_effective_sample_size" in summary

    written = json.loads((tmp_path / "mass_sampling.json").read_text())
    assert written == payload


def test_sampling_study_replays_byte_identically(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    run_mass_sampling_study(DECISION_CONFIG, SAMPLING_CONFIG, first)
    run_mass_sampling_study(DECISION_CONFIG, SAMPLING_CONFIG, second)

    assert (first / "mass_sampling.json").read_bytes() == (
        second / "mass_sampling.json"
    ).read_bytes()
