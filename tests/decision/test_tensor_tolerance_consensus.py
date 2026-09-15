from __future__ import annotations

import json
from pathlib import Path

from qldpc_fno.decision.tensor_consensus import _apply_policy
from qldpc_fno.decision.tensor_policy_data import generate_tensor_policy_data
from qldpc_fno.decision.tensor_tolerance_consensus import (
    _CANONICAL_DATA_CONFIG,
    _CANONICAL_POLICY,
    _action_metadata_integrity,
    _generation_provenance_integrity,
    _outcome_replay_integrity,
    run_tensor_tolerance_consensus_study,
)
from tests.decision.test_tensor_policy_data import _config as base_data_config


def _reduced_data_config(tmp_path: Path) -> Path:
    payload = json.loads(base_data_config(tmp_path).read_text())
    payload["actions"] = [
        {"id": "columns_tol001", "mode": "columns", "tol": 0.01},
        {"id": "rows_tol001", "mode": "rows", "tol": 0.01},
        {"id": "columns_chi8", "mode": "columns", "chi": 8},
        {"id": "rows_chi8", "mode": "rows", "chi": 8},
    ]
    payload["work_trace_detail"] = "aggregate"
    path = tmp_path / "reduced-tolerance-data.json"
    path.write_text(json.dumps(payload))
    return path


def test_tracked_tolerance_policy_and_data_contract_are_exact() -> None:
    assert json.loads(Path("configs/tensor_tolerance_consensus_policy.json").read_text()) == (
        _CANONICAL_POLICY
    )
    assert json.loads(Path("configs/tensor_tolerance_confirmation_data.json").read_text()) == (
        _CANONICAL_DATA_CONFIG
    )


def test_reduced_tolerance_consensus_is_scored_but_not_confirmed(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    generate_tensor_policy_data(_reduced_data_config(tmp_path), data_dir)

    payload = run_tensor_tolerance_consensus_study(
        Path("configs/tensor_tolerance_consensus_policy.json"),
        data_dir / "tensor_policy_data.json",
        tmp_path / "result",
    )

    assert payload["run_label"] == "reduced_nonconfirmatory"
    assert payload["gate5_tolerance_confirmation_status"] == (
        "unresolved_noncanonical_data"
    )
    assert payload["statistical_contract"]["endpoint"] == (
        "exact-reference logical-class agreement only"
    )
    assert payload["work"]["latency_claim"] is False
    assert payload["posterior_mass_fidelity_claim"] is False
    assert (tmp_path / "result" / "tensor_tolerance_consensus.json").exists()


def _synthetic_context(*, column_class: int | None, row_class: int | None) -> dict[str, object]:
    def outcome(action_id: str, selected: int | None, work: int) -> dict[str, object]:
        return {
            "action_id": action_id,
            "solver_valid": selected is not None,
            "selected_class": selected,
            "selected_class_correct": selected == 1,
            "decision_regret": 0.0 if selected == 1 else 0.1,
            "mass_fidelity_feasible": False,
            "work": {"estimated_arithmetic_flops": work},
        }

    return {
        "distance": 5,
        "outcomes": [
            outcome("columns_tol001", column_class, 11),
            outcome("rows_tol001", row_class, 13),
            outcome("columns_chi8", 1, 37),
        ],
    }


def test_policy_executes_both_views_then_fallback_only_on_disagreement_or_invalidity() -> None:
    identity = {key: key for key in ("columns_tol001", "rows_tol001", "columns_chi8")}
    agreed = _apply_policy(_synthetic_context(column_class=1, row_class=1), _CANONICAL_POLICY, identity)
    disagreed = _apply_policy(_synthetic_context(column_class=0, row_class=1), _CANONICAL_POLICY, identity)
    invalid = _apply_policy(_synthetic_context(column_class=None, row_class=1), _CANONICAL_POLICY, identity)

    assert agreed["executed_actions"] == ["columns_tol001", "rows_tol001"]
    assert agreed["estimated_arithmetic_flops"] == 24
    assert agreed["fallback_used"] is False
    for result in (disagreed, invalid):
        assert result["executed_actions"] == [
            "columns_tol001", "rows_tol001", "columns_chi8"
        ]
        assert result["estimated_arithmetic_flops"] == 61
        assert result["fallback_used"] is True
        assert result["selected_class_correct"] is True


def test_paired_work_tamper_and_stale_source_provenance_are_rejected(tmp_path: Path) -> None:
    payload = generate_tensor_policy_data(_reduced_data_config(tmp_path), tmp_path / "tamper-data")

    assert _action_metadata_integrity(payload)
    assert _generation_provenance_integrity(payload)
    original = next(row for row in payload["contexts"] if row["orientation"] == "original")
    transpose = next(row for row in payload["contexts"] if row["orientation"] == "transpose")
    for context in (original, transpose):
        outcome = next(row for row in context["outcomes"] if row["action_id"] == "columns_chi8")
        outcome["work"]["estimated_arithmetic_flops"] += 10
    assert not _action_metadata_integrity(payload)
    payload["provenance"]["source_sha256"][next(iter(payload["provenance"]["source_sha256"]))] = "0" * 64
    assert not _generation_provenance_integrity(payload)


def test_paired_correctness_label_tamper_is_rejected(tmp_path: Path) -> None:
    payload = generate_tensor_policy_data(_reduced_data_config(tmp_path), tmp_path / "label-data")
    assert _action_metadata_integrity(payload)
    for context in payload["contexts"]:
        for outcome in context["outcomes"]:
            outcome["selected_class_correct"] = not outcome["selected_class_correct"]
    assert not _action_metadata_integrity(payload)


def test_replay_rejects_fabricated_reference_or_action_probabilities(tmp_path: Path) -> None:
    payload = generate_tensor_policy_data(_reduced_data_config(tmp_path), tmp_path / "replay-data")
    assert _outcome_replay_integrity(payload)
    payload["contexts"][0]["reference_probabilities"][0] += 0.01
    assert not _outcome_replay_integrity(payload)
    payload["contexts"][0]["reference_probabilities"][0] -= 0.01
    outcome = payload["contexts"][0]["outcomes"][0]
    outcome["probabilities"][0] += 0.01
    assert not _outcome_replay_integrity(payload)


def test_replay_rejects_fabricated_reference_audit_metrics(tmp_path: Path) -> None:
    payload = generate_tensor_policy_data(_reduced_data_config(tmp_path), tmp_path / "audit-data")
    assert _outcome_replay_integrity(payload)
    context = payload["contexts"][0]
    context["reference_row_column_log_ratio_difference"] = 1.0
    assert not _outcome_replay_integrity(payload)
    context["reference_row_column_log_ratio_difference"] = 0.0
    context["reference_enumeration_probability_difference"] = 1.0
    assert not _outcome_replay_integrity(payload)
