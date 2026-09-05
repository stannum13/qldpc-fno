from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.temporal.config import CausalExperimentConfig
from qldpc_fno.temporal.screen import (
    generate_sequence_campaign,
    publish_screen_result,
    recompute_screen_result,
    verify_screen_result,
    verify_sequence_campaign,
)
from qldpc_fno.temporal.seeds import REGIMES

CONFIG_PATH = Path("configs/causal_fno_hippo_reduced.json")


def _tiny_config(path: Path) -> Path:
    payload = json.loads(CONFIG_PATH.read_text())
    payload["splits"] = {"train": 1, "validation": 1, "calibration": 1, "test": 0}
    payload["rounds"] = {"burn_in": 1, "scored": 1}
    write_canonical_json(path, payload)
    return path


def _scientific_payload() -> dict[str, object]:
    round_row = {
        "sequence_id": "a" * 64,
        "round": 1,
        "logical_failure": False,
        "syndrome_valid": True,
        "converged": True,
        "iterations": 2,
        "correction_weight": 0,
        "predicted_observables": [0],
        "true_logical_flips": [0],
        "observed_error_weight": 0,
        "forecast_nll": 0.1,
        "forecast_brier": 0.01,
    }
    arm_names = (
        "stationary_field",
        "ewma",
        "logistic_ar",
        "cnn_fir",
        "fno_fir",
        "cnn_hippo",
        "fno_hippo",
    )

    def arm(regime: str, name: str) -> dict[str, object]:
        return {
            "name": name,
            "predictor_type": name,
            "regime": regime,
            "role": "validation",
            "parameter_count": {"stored": 3, "effective": 3},
            "forecast": {
                "overall_nll": 0.1,
                "overall_brier": 0.01,
                "ece": 0.0,
                "reliability": [],
            },
            "decoder": {
                "overall_bler": 0.0,
                "convergence_rate": 1.0,
                "mean_iterations": 2.0,
                "mean_correction_weight": 0.0,
                "all_syndrome_valid": True,
            },
            "hashes": {
                "partition": "b" * 64,
                "partition_content": "c" * 64,
                "provenance": "d" * 64,
                "evaluation_content": "6" * 64,
                "hx": "e" * 64,
                "hz": "f" * 64,
                "logical_x": "1" * 64,
            },
            "sequence_summaries": [
                {
                    "sequence_id": "a" * 64,
                    "nll": 0.1,
                    "brier": 0.01,
                    "bler": 0.0,
                }
            ],
            "per_round_outcomes": [round_row],
        }

    regimes = {
        regime: {
            "selected_observation_baseline": "ewma",
            "baseline_validation_nll": {
                "stationary_field": 0.1,
                "ewma": 0.1,
                "logistic_ar": 0.1,
            },
            "hashes": {
                "decoder_config": "4" * 64,
                "partition": "b" * 64,
                "partition_content": "c" * 64,
                "baseline_fit_policy": "5" * 64,
            },
            "arms": {name: arm(regime, name) for name in arm_names},
        }
        for regime in REGIMES
    }
    progressions = {
        regime: {
            "selected_name": "ewma",
            "nll_improvement": 0.0,
            "bler_difference": 0.0,
            "progressed": False,
            "scope": "descriptive_reduced_non_scientific",
            "p_value": None,
            "hypothesis_status": None,
        }
        for regime in REGIMES
        if regime != "stationary_iid"
    }
    audit_row = {
        "passed": True,
        "forecast_round": 1,
        "checks": [
            {"name": name, "bit_identical": True}
            for name in (
                "current_future_syndromes",
                "physical_error_labels",
                "logical_labels",
                "privileged_diagnostics",
            )
        ],
    }
    return {
        "artifact_mode": "reduced_non_scientific",
        "scientific_scope": "reduced_non_scientific",
        "p_value": None,
        "hypothesis_status": None,
        "source": {
            "config_sha256": "2" * 64,
            "sequences_manifest_sha256": "3" * 64,
            "source_commit": "4" * 40,
        },
        "causal_audit": {
            "passed": True,
            "arms": {regime: {name: audit_row for name in arm_names[3:]} for regime in REGIMES},
            "overfit_fixture": {
                name: {"steps": 1, "nll": 0.01, "accuracy": 1.0} for name in arm_names[3:]
            },
        },
        "regimes": regimes,
        "progressions": progressions,
        "factor_diagnostics": {
            "in_basis_interaction": [0.0],
            "basis_mismatch_interaction": [0.0],
            "in_basis_mean": 0.0,
            "basis_mismatch_mean": 0.0,
            "in_basis_supports_predeclared_direction": False,
            "basis_mismatch_retains_predeclared_direction": False,
            "scope": "descriptive_reduced_non_scientific",
            "p_value": None,
            "hypothesis_status": None,
        },
    }


def test_generate_and_directly_regenerate_all_a_to_e_role_artifacts(tmp_path: Path) -> None:
    config_path = _tiny_config(tmp_path / "config.json")
    output = tmp_path / "sequences"

    first = generate_sequence_campaign(config_path=config_path, output_dir=output)
    verify_sequence_campaign(config_path=config_path, output_dir=output, regenerate=True)
    second = generate_sequence_campaign(config_path=config_path, output_dir=output)

    assert first == second
    assert len(first["sequences"]) == 15
    assert (output / "manifest.json").is_file()
    assert list(dict.fromkeys(row["regime"] for row in first["sequences"])) == list(
        CausalExperimentConfig.from_json(config_path).regimes
    )
    assert {row["role"] for row in first["sequences"]} == {
        "train",
        "validation",
        "calibration",
    }

    changed = json.loads(config_path.read_text())
    changed["campaign_seed"] += 1
    write_canonical_json(config_path, changed)
    with pytest.raises(FileExistsError, match="completed differing"):
        generate_sequence_campaign(config_path=config_path, output_dir=output)


def test_screen_publication_is_deterministic_and_recomputable(tmp_path: Path) -> None:
    payload = _scientific_payload()
    first = tmp_path / "screen-a"
    second = tmp_path / "screen-b"

    publish_screen_result(first, payload, timing={"wall_seconds": 1.0})
    publish_screen_result(second, payload, timing={"wall_seconds": 9.0})
    verify_screen_result(first)
    verify_screen_result(second)

    assert (first / "results.json").read_bytes() == (second / "results.json").read_bytes()
    assert sha256_file(first / "results.json") == sha256_file(second / "results.json")
    assert (first / "timing.json").read_bytes() != (second / "timing.json").read_bytes()
    assert (first / "manifest.json").is_file()

    differing = json.loads(json.dumps(payload))
    differing["source"]["config_sha256"] = "9" * 64
    with pytest.raises(FileExistsError, match="completed differing"):
        publish_screen_result(first, differing, timing={})

    tampered = json.loads(json.dumps(payload))
    tampered["regimes"]["joint_in_basis"]["arms"]["stationary_field"]["forecast"]["overall_nll"] = (
        9.0
    )
    with pytest.raises(ValueError, match="recomputation"):
        recompute_screen_result(tampered)


def test_result_verifier_rejects_incomplete_experiment_matrix() -> None:
    payload = _scientific_payload()
    del payload["regimes"]["temporal_uniform"]

    with pytest.raises(ValueError, match="exact A-E"):
        recompute_screen_result(payload)

    payload = _scientific_payload()
    del payload["regimes"]["stationary_iid"]["arms"]["fno_hippo"]
    with pytest.raises(ValueError, match="exact arm set"):
        recompute_screen_result(payload)


@pytest.mark.parametrize(
    ("script", "command"),
    [
        ("experiments/19_generate_causal_sequences.py", "verify"),
        ("experiments/20_run_causal_factor_screen.py", "verify"),
    ],
)
def test_causal_cli_entrypoints_expose_verification_commands(script: str, command: str) -> None:
    result = subprocess.run(
        [sys.executable, script, command, "--help"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--out" in result.stdout
