from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

from qldpc_fno.artifacts import sha256_file, write_canonical_json


def _write_noncanonical_code(path: Path) -> None:
    path.mkdir()
    hx_path = path / "hx.npz"
    hz_path = path / "hz.npz"
    sparse.save_npz(hx_path, sparse.csr_matrix([[1, 1]], dtype=np.uint8))
    sparse.save_npz(hz_path, sparse.csr_matrix((0, 2), dtype=np.uint8))
    write_canonical_json(
        path / "code.json",
        {
            "ell": 2,
            "hx_sha256": sha256_file(hx_path),
            "hz_sha256": sha256_file(hz_path),
            "k": 1,
            "n": 2,
            "name": "tiny",
        },
    )


def _write_reduced_config(path: Path, *, training_seed: int = 1701) -> None:
    write_canonical_json(
        path,
        {
            "campaign_seed": 20260901,
            "noise_grid": [0.003, 0.005],
            "selection_mode": "pilot",
            "pilot_shots_per_point": 8,
            "train_shots_cap": 8,
            "calibration_shots_cap": 8,
            "calibration_decode_shots_cap": 8,
            "calibration_shortlist_per_method": 1,
            "test_batch_shots": 8,
            "max_test_shots_per_point": 8,
            "target_failures": 4,
            "test_stopping_mode": "adaptive",
            "training_epochs": 1,
            "training_batch_size": 1,
            "training_learning_rate": 0.001,
            "training_seed": training_seed,
            "checkpoint_every_epochs": 1,
            "cloud_cpu": 1,
            "cloud_memory": "1Gi",
            "cloud_timeout_seconds": 3600,
            "checkpoint_grace_seconds": 2700,
        },
    )


def _write_flow_config(path: Path) -> None:
    write_canonical_json(
        path,
        {
            "campaign_seed": 20260901,
            "noise_grid": [0.003],
            "selection_mode": "pilot",
            "pilot_shots_per_point": 1,
            "train_shots_cap": 16,
            "calibration_shots_cap": 16,
            "calibration_decode_shots_cap": 16,
            "calibration_shortlist_per_method": 1,
            "test_batch_shots": 1,
            "max_test_shots_per_point": 1,
            "target_failures": 1,
            "test_stopping_mode": "adaptive",
            "training_epochs": 1,
            "training_batch_size": 1,
            "training_learning_rate": 0.001,
            "training_seed": 1701,
            "checkpoint_every_epochs": 1,
            "cloud_cpu": 1,
            "cloud_memory": "1Gi",
            "cloud_timeout_seconds": 3600,
            "checkpoint_grace_seconds": 2700,
        },
    )


def _write_selection(path: Path, *, config: Path, code_manifest: Path) -> None:
    write_canonical_json(
        path,
        {
            "pilot_rows": [],
            "selected_noise_points": [0.003, 0.005],
            "source_sha256": {
                "code_manifest": sha256_file(code_manifest),
                "config": sha256_file(config),
            },
        },
    )
    write_canonical_json(
        path.parent / "manifest.json",
        {"complete": True, "role": "pilot", "selection_sha256": sha256_file(path)},
    )


def _run_shard_cli(
    *,
    config: Path,
    code: Path,
    selection: Path,
    output: Path,
    role: str = "train",
    shots: int = 8,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "experiments/14_generate_campaign_shards.py",
            "--config",
            str(config),
            "--code",
            str(code),
            "--selection",
            str(selection),
            "--role",
            role,
            "--shots-per-rate",
            str(shots),
            "--out",
            str(output),
        ],
        capture_output=True,
        check=False,
        text=True,
    )


def test_reduced_real_code_campaign_data_flow_with_dynamic_extension(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    config_path = tmp_path / "campaign.json"
    campaign_dir = tmp_path / "campaign"
    pilot_dir = campaign_dir / "pilot"
    _write_flow_config(config_path)
    subprocess.run(
        [sys.executable, "experiments/01_build_lp_codes.py", "--out", str(code_dir)],
        check=True,
    )
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
    assert len(selection["pilot_rows"]) > 1
    assert selection["pilot_rows"][1]["error_rate"] == 0.0045
    assert json.loads((pilot_dir / "manifest.json").read_text())["complete"] is True

    for role, shots in (("train", 16), ("calibration", 16), ("test", 1)):
        result = _run_shard_cli(
            config=config_path,
            code=code_dir,
            selection=selection_path,
            output=campaign_dir / role,
            role=role,
            shots=shots,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads((campaign_dir / role / "manifest.json").read_text())["complete"] is True

    role_shots = {}
    for role in ("train", "calibration", "test"):
        manifests = [
            json.loads(path.read_text())
            for path in (campaign_dir / role).glob("rate-*/shard-*/samples.json")
        ]
        role_shots[role] = sum(int(manifest["shots"]) for manifest in manifests)
    assert role_shots["train"] == 16
    assert role_shots["calibration"] == 16
    assert role_shots["test"] == len(selection["selected_noise_points"])


def test_pilot_cli_rejects_noncanonical_code(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    config_path = tmp_path / "campaign.json"
    _write_noncanonical_code(code_dir)
    _write_reduced_config(config_path)

    result = subprocess.run(
        [
            sys.executable,
            "experiments/13_pilot_noise_grid.py",
            "--config",
            str(config_path),
            "--code",
            str(code_dir),
            "--out",
            str(tmp_path / "pilot"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode != 0
    assert "canonical lp_3_7_16" in result.stderr


def test_shard_cli_rejects_noncanonical_code(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    config_path = tmp_path / "campaign.json"
    pilot_dir = tmp_path / "pilot"
    pilot_dir.mkdir()
    selection_path = pilot_dir / "selection.json"
    _write_noncanonical_code(code_dir)
    _write_reduced_config(config_path)
    _write_selection(selection_path, config=config_path, code_manifest=code_dir / "code.json")

    result = _run_shard_cli(
        config=config_path,
        code=code_dir,
        selection=selection_path,
        output=tmp_path / "train",
    )

    assert result.returncode != 0
    assert "canonical lp_3_7_16" in result.stderr


def test_shard_cli_rejects_corrupted_selection(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    config_path = tmp_path / "campaign.json"
    pilot_dir = tmp_path / "pilot"
    pilot_dir.mkdir()
    selection_path = pilot_dir / "selection.json"
    _write_noncanonical_code(code_dir)
    _write_reduced_config(config_path)
    _write_selection(selection_path, config=config_path, code_manifest=code_dir / "code.json")
    selection = json.loads(selection_path.read_text())
    selection["selected_noise_points"].append(0.008)
    selection_path.write_text(json.dumps(selection))

    result = _run_shard_cli(
        config=config_path,
        code=code_dir,
        selection=selection_path,
        output=tmp_path / "train",
    )

    assert result.returncode != 0
    assert "selection SHA-256 mismatch" in result.stderr


def test_shard_cli_rejects_cross_config_selection(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    source_config = tmp_path / "source.json"
    current_config = tmp_path / "current.json"
    pilot_dir = tmp_path / "pilot"
    pilot_dir.mkdir()
    selection_path = pilot_dir / "selection.json"
    _write_noncanonical_code(code_dir)
    _write_reduced_config(source_config, training_seed=1701)
    _write_reduced_config(current_config, training_seed=1702)
    _write_selection(selection_path, config=source_config, code_manifest=code_dir / "code.json")

    result = _run_shard_cli(
        config=current_config,
        code=code_dir,
        selection=selection_path,
        output=tmp_path / "train",
    )

    assert result.returncode != 0
    assert "campaign configuration SHA-256 mismatch" in result.stderr


def test_shard_cli_rejects_cross_code_selection(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    config_path = tmp_path / "campaign.json"
    pilot_dir = tmp_path / "pilot"
    pilot_dir.mkdir()
    selection_path = pilot_dir / "selection.json"
    _write_noncanonical_code(code_dir)
    _write_reduced_config(config_path)
    _write_selection(selection_path, config=config_path, code_manifest=code_dir / "code.json")
    code_manifest = json.loads((code_dir / "code.json").read_text())
    code_manifest["name"] = "changed"
    write_canonical_json(code_dir / "code.json", code_manifest)

    result = _run_shard_cli(
        config=config_path,
        code=code_dir,
        selection=selection_path,
        output=tmp_path / "train",
    )

    assert result.returncode != 0
    assert "code manifest SHA-256 mismatch" in result.stderr
