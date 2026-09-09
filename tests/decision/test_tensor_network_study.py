from __future__ import annotations

import hashlib
import json
from pathlib import Path

from qldpc_fno.decision.tensor_network_study import run_tensor_network_study


def _config(tmp_path: Path) -> Path:
    domain = "qldpc-fno/tensor-network-test/v1"
    payload = {
        "schema_version": 1,
        "seed_domain": domain,
        "campaign_seed": int.from_bytes(hashlib.sha256(domain.encode()).digest()[:8], "big"),
        "code_distances": [3],
        "error_rates": [0.1],
        "instances_per_point": 2,
        "chi_ladder": [1, 2],
        "contraction_modes": ["columns"],
        "unrestricted_reference_max_distance": 3,
        "high_chi_reference": 4,
        "timing_warmups": 0,
        "timing_repetitions": 1,
        "log_mass_ratio_tolerance": 0.05,
        "log_mass_ratio_tolerance_sweep": [0.01, 0.05],
        "minimum_oracle_work_savings_fraction": 0.05,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return path


def test_study_uses_paired_syndromes_and_explicit_reference_labels(tmp_path: Path) -> None:
    payload = run_tensor_network_study(_config(tmp_path), tmp_path / "out")

    assert payload["run_label"] == "reduced_non_scientific"
    assert payload["gate5_status"] == "unresolved_reduced_non_scientific"
    assert len(payload["instances"]) == 2
    assert len(payload["rows"]) == 2 * 2
    assert {row["reference_kind"] for row in payload["rows"]} == {
        "unrestricted_mps_exact_up_to_roundoff"
    }
    assert all(row["work"]["pairwise_contractions"] > 0 for row in payload["rows"])
    assert all(len(row["untraced_wall_seconds"]) == 1 for row in payload["rows"])
    assert all(row["reference_selected_class"] in range(4) for row in payload["rows"])
    assert all(row["selected_class"] in range(4) for row in payload["rows"])
    assert payload["monotonicity_audit"]["tested_sequences"] == 2
    assert "best_fixed_action" in payload["adaptive_signal"]
    assert "per_instance_oracle" in payload["adaptive_signal"]
    assert [row["tolerance"] for row in payload["criterion_sensitivity"]] == [0.01, 0.05]
    assert (tmp_path / "out" / "tensor_network_reference.json").exists()
