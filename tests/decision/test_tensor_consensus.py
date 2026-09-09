from __future__ import annotations

import json
from pathlib import Path

from qldpc_fno.decision.tensor_consensus import (
    _CANONICAL_POLICY,
    _apply_policy,
    _data_integrity,
    _load_policy,
    run_tensor_consensus_study,
    wilson_upper,
)
from qldpc_fno.decision.tensor_policy_data import generate_tensor_policy_data
from tests.decision.test_tensor_policy_data import _config as data_config


def _policy_config(tmp_path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "policy_id": "two_view_chi4_consensus_v1",
        "direct_action_by_distance": {"3": "columns_chi4"},
        "consensus_by_distance": {
            "5": {
                "actions": ["columns_chi4", "rows_chi4"],
                "fallback": "columns_chi8",
            }
        },
        "fixed_fallback_action": "columns_chi8",
        "required_data_seed_domain": "fresh-domain-not-the-reduced-test",
        "required_confirmation_per_stratum": 160,
        "familywise_alpha": 0.05,
        "familywise_comparisons": 12,
        "maximum_unsafe_rate": 0.05,
        "minimum_work_savings_fraction": 0.05,
        "work_savings_confidence_alpha": 0.05,
        "work_savings_bootstrap_replicates": 100,
        "work_savings_bootstrap_seed": 17,
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload))
    return path


def test_zero_failure_wilson_bound_meets_frozen_confirmation_gate() -> None:
    adjusted_alpha = 0.05 / 12

    assert wilson_upper(0, 160, alpha=adjusted_alpha) < 0.05
    assert wilson_upper(1, 160, alpha=adjusted_alpha) > 0.05


def test_reduced_data_evaluates_but_cannot_open_confirmation_gate(tmp_path: Path) -> None:
    data_path = tmp_path / "data"
    generate_tensor_policy_data(data_config(tmp_path), data_path)

    payload = run_tensor_consensus_study(
        _policy_config(tmp_path),
        data_path / "tensor_policy_data.json",
        tmp_path / "out",
    )

    assert payload["run_label"] == "reduced_nonconfirmatory"
    assert payload["gate5_confirmation_status"] == "unresolved_noncanonical_data"
    assert len(payload["decision_rows"]) == 1
    assert payload["decision_rows"][0]["independent_unit"] == "base_group_original_orientation"
    assert payload["transpose_consistency"]["tested_group_count"] == 1
    assert (tmp_path / "out" / "tensor_consensus.json").exists()


def test_invalid_fallback_is_an_explicit_policy_failure(tmp_path: Path) -> None:
    def outcome(action_id: str, selected: int | None, valid: bool) -> dict[str, object]:
        return {
            "action_id": action_id,
            "solver_valid": valid,
            "selected_class": selected,
            "selected_class_correct": False,
            "decision_regret": None,
            "mass_fidelity_feasible": False,
            "work": {"estimated_arithmetic_flops": 10},
        }

    context = {
        "distance": 5,
        "outcomes": [
            outcome("columns_chi4", 0, True),
            outcome("rows_chi4", 1, True),
            outcome("columns_chi8", None, False),
        ],
    }
    policy = _policy_config(tmp_path).read_text()
    result = _apply_policy(
        context,
        json.loads(policy),
        {
            "columns_chi4": "columns_chi4",
            "rows_chi4": "rows_chi4",
            "columns_chi8": "columns_chi8",
        },
    )

    assert result["fallback_used"] is True
    assert result["selected_solver_valid"] is False
    assert result["selected_class"] is None
    assert result["selected_class_correct"] is False


def test_tracked_policy_is_the_exact_frozen_canonical_policy() -> None:
    assert _load_policy(Path("configs/tensor_consensus_policy.json")) == _CANONICAL_POLICY


def test_integrity_replays_seeds_and_rejects_relabelled_draws(tmp_path: Path) -> None:
    output = tmp_path / "data-integrity"
    payload = generate_tensor_policy_data(data_config(tmp_path), output)

    assert _data_integrity(payload, payload["config"], payload["transpose_action_map"])
    payload["base_instances"][0]["sampler_seed"] += 1
    assert not _data_integrity(payload, payload["config"], payload["transpose_action_map"])
