from __future__ import annotations

import hashlib
import json
from pathlib import Path

from qldpc_fno.decision.tensor_policy_data import generate_tensor_policy_data


def _config(tmp_path: Path) -> Path:
    domain = "qldpc-fno/tensor-policy-data-test/v1"
    payload = {
        "schema_version": 1,
        "seed_domain": domain,
        "campaign_seed": int.from_bytes(hashlib.sha256(domain.encode()).digest()[:8], "big"),
        "code_distances": [3],
        "error_rates": [0.1],
        "base_instances_per_stratum": {"train": 1, "calibration": 1, "confirmation": 1},
        "actions": [
            {"id": "columns_chi4", "mode": "columns", "chi": 4},
            {"id": "rows_chi4", "mode": "rows", "chi": 4},
            {"id": "columns_chi8", "mode": "columns", "chi": 8},
            {"id": "rows_chi8", "mode": "rows", "chi": 8},
        ],
        "mass_log_ratio_tolerance": 0.05,
        "reference_probability_tolerance": 1e-10,
        "reference_log_ratio_tolerance": 1e-8,
        "enumerate_distance3": True,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return path


def test_policy_data_has_grouped_transposes_and_no_reference_features(tmp_path: Path) -> None:
    payload = generate_tensor_policy_data(_config(tmp_path), tmp_path / "out")

    assert len(payload["base_instances"]) == 3
    assert len(payload["contexts"]) == 6
    assert len({row["sampler_seed"] for row in payload["base_instances"]}) == 3
    assert payload["reference_audit"]["invalid_context_count"] == 0
    assert payload["reference_audit"]["maximum_row_column_probability_difference"] < 1e-10
    assert payload["reference_audit"]["maximum_enumeration_probability_difference"] < 1e-10
    assert payload["reference_audit"]["maximum_row_column_log_ratio_difference"] < 1e-8
    assert payload["reference_audit"]["maximum_enumeration_log_ratio_difference"] < 1e-8
    assert payload["transpose_audit"]["solver_validity_mismatch_count"] == 0
    assert payload["transpose_audit"]["mass_fidelity_feasibility_mismatch_count"] == 0

    for base in payload["base_instances"]:
        pair = [row for row in payload["contexts"] if row["group_id"] == base["group_id"]]
        assert len(pair) == 2
        assert {row["orientation"] for row in pair} == {"original", "transpose"}
        assert {row["split"] for row in pair} == {base["split"]}
        assert all(len(row["outcomes"]) == 4 for row in pair)
        assert all("reference_selected_class" not in row["feature_names"] for row in pair)
        assert all("physical_error" not in row for row in pair)

    swaps = payload["transpose_action_map"]
    assert swaps["columns_chi4"] == "rows_chi4"
    assert swaps["rows_chi8"] == "columns_chi8"


def test_policy_data_replays_byte_identically(tmp_path: Path) -> None:
    config = _config(tmp_path)

    generate_tensor_policy_data(config, tmp_path / "first")
    generate_tensor_policy_data(config, tmp_path / "second")

    assert (tmp_path / "first" / "tensor_policy_data.json").read_bytes() == (
        tmp_path / "second" / "tensor_policy_data.json"
    ).read_bytes()


def test_canonical_action_set_retains_the_discovery_winners() -> None:
    payload = json.loads(Path("configs/tensor_policy_data.json").read_text())

    assert {action["id"] for action in payload["actions"]} >= {
        "columns_chi2",
        "rows_chi2",
        "columns_chi4",
        "rows_chi4",
        "average_chi4",
        "columns_chi8",
    }


def test_confirmation_only_config_does_not_invent_development_draws(tmp_path: Path) -> None:
    config = json.loads(_config(tmp_path).read_text())
    config["base_instances_per_stratum"] = {
        "train": 0,
        "calibration": 0,
        "confirmation": 1,
    }
    path = tmp_path / "confirmation-only.json"
    path.write_text(json.dumps(config))

    payload = generate_tensor_policy_data(path, tmp_path / "confirmation-only")

    assert len(payload["base_instances"]) == 1
    assert {row["split"] for row in payload["contexts"]} == {"confirmation"}
