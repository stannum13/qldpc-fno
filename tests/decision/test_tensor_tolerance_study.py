from __future__ import annotations

import json
from pathlib import Path

from qldpc_fno.artifacts import sha256_file
from qldpc_fno.decision.tensor_network_study import run_tensor_network_study
from qldpc_fno.decision.tensor_tolerance_study import run_tensor_tolerance_study
from tests.decision.test_tensor_network_study import _config as reference_config


def test_tolerance_study_uses_locked_reference_and_records_local_spectra(
    tmp_path: Path,
) -> None:
    reference_dir = tmp_path / "reference"
    run_tensor_network_study(reference_config(tmp_path), reference_dir)
    reference_path = reference_dir / "tensor_network_reference.json"
    config = {
        "schema_version": 1,
        "expected_reference_sha256": sha256_file(reference_path),
        "tolerances": [0.001, 0.01],
        "contraction_modes": ["columns", "rows"],
        "logical_class_required": True,
        "log_mass_ratio_tolerance": 0.05,
        "fixed_comparator": {"mode": "columns", "chi": 2},
    }
    config_path = tmp_path / "tolerance.json"
    config_path.write_text(json.dumps(config))

    payload = run_tensor_tolerance_study(
        config_path, reference_path, tmp_path / "tolerance-out"
    )

    assert payload["run_label"] == "reduced_non_scientific"
    assert len(payload["rows"]) == 2 * 2 * 2
    assert payload["reference_sha256"] == sha256_file(reference_path)
    assert all(row["chi"] is None for row in payload["rows"])
    assert all(row["truncation_event_count"] > 0 for row in payload["rows"])
    assert all(row["retained_rank_max"] >= row["retained_rank_min"] for row in payload["rows"])
    assert all(
        0 <= row["maximum_local_discarded_squared_weight_fraction"] <= 1
        for row in payload["rows"]
    )
    assert payload["fixed_comparator"]["mode"] == "columns"
    assert "best_globally_class_safe_tolerance_action" in payload
    assert (tmp_path / "tolerance-out" / "tensor_tolerance_discovery.json").exists()
