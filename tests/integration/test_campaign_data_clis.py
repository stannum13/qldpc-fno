from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

from qldpc_fno.artifacts import sha256_file


def _write_tiny_code(path: Path) -> None:
    path.mkdir()
    hx_path = path / "hx.npz"
    hz_path = path / "hz.npz"
    sparse.save_npz(hx_path, sparse.csr_matrix([[1, 1]], dtype=np.uint8))
    sparse.save_npz(hz_path, sparse.csr_matrix((0, 2), dtype=np.uint8))
    (path / "code.json").write_text(
        json.dumps(
            {
                "hx_sha256": sha256_file(hx_path),
                "hz_sha256": sha256_file(hz_path),
                "name": "tiny",
            }
        )
    )


def _write_reduced_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "campaign_seed": 20260901,
                "noise_grid": [0.1, 0.2],
                "pilot_shots_per_point": 8,
                "train_shots_cap": 8,
                "calibration_shots_cap": 8,
                "test_batch_shots": 8,
                "max_test_shots_per_point": 8,
                "target_failures": 4,
                "training_epochs": 1,
                "training_batch_size": 1,
                "training_learning_rate": 0.001,
                "training_seed": 1701,
                "checkpoint_every_epochs": 1,
                "cloud_cpu": 1,
                "cloud_memory": "1Gi",
                "cloud_timeout_seconds": 60,
                "checkpoint_grace_seconds": 1,
            }
        )
    )


def test_pilot_and_role_shard_clis_publish_provenance(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    config_path = tmp_path / "campaign.json"
    campaign_dir = tmp_path / "campaign"
    pilot_dir = campaign_dir / "pilot"
    _write_tiny_code(code_dir)
    _write_reduced_config(config_path)

    subprocess.run(
        [
            sys.executable,
            "experiments/13_pilot_noise_grid.py",
            "--config",
            str(config_path),
            "--code",
            str(code_dir),
            "--out",
            str(pilot_dir),
        ],
        check=True,
    )
    selection_path = pilot_dir / "selection.json"
    selection = json.loads(selection_path.read_text())
    assert len(selection["pilot_rows"]) == 2
    assert selection["selected_noise_points"] == sorted(selection["selected_noise_points"])
    for row in selection["pilot_rows"]:
        assert row["shots"] == 8
        assert 0 <= row["block_errors"] <= 8
        assert 0 <= row["syndrome_valid"] <= 8
        assert 0 <= row["converged"] <= 8
        assert row["latency_seconds"] >= 0

    for role in ("train", "calibration", "test"):
        subprocess.run(
            [
                sys.executable,
                "experiments/14_generate_campaign_shards.py",
                "--config",
                str(config_path),
                "--code",
                str(code_dir),
                "--selection",
                str(selection_path),
                "--role",
                role,
                "--shots-per-rate",
                "8",
                "--out",
                str(campaign_dir / role),
            ],
            check=True,
        )

    manifests = [
        json.loads(path.read_text())
        for path in campaign_dir.glob("*/rate-*/shard-*/samples.json")
    ]
    assert {manifest["role"] for manifest in manifests} == {
        "pilot",
        "train",
        "calibration",
        "test",
    }
    assert len({manifest["seed"] for manifest in manifests}) == len(manifests)
    code_hash = sha256_file(code_dir / "code.json")
    assert all(manifest["source_sha256"]["code_manifest"] == code_hash for manifest in manifests)
    for manifest in manifests:
        assert manifest["source_sha256"]["config"] == sha256_file(config_path)
        if manifest["role"] != "pilot":
            assert manifest["source_sha256"]["selection"] == sha256_file(selection_path)


def test_shard_cli_refuses_existing_completion_manifest(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    config_path = tmp_path / "campaign.json"
    selection_path = tmp_path / "selection.json"
    output_dir = tmp_path / "train"
    _write_tiny_code(code_dir)
    _write_reduced_config(config_path)
    selection_path.write_text(json.dumps({"selected_noise_points": [0.1]}))
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text(json.dumps({"complete": True}))

    result = subprocess.run(
        [
            sys.executable,
            "experiments/14_generate_campaign_shards.py",
            "--config",
            str(config_path),
            "--code",
            str(code_dir),
            "--selection",
            str(selection_path),
            "--role",
            "train",
            "--shots-per-rate",
            "8",
            "--out",
            str(output_dir),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode != 0
    assert "completion manifest" in result.stderr
