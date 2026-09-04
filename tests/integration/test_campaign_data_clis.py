from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.campaign.shards import select_noise_points


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


def _write_reduced_config(
    path: Path,
    *,
    training_seed: int = 1701,
    selection_mode: str = "fixed",
) -> None:
    write_canonical_json(
        path,
        {
            "campaign_seed": 20260901,
            "noise_grid": [0.003, 0.005],
            "selection_mode": selection_mode,
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


def _write_fixed_config(path: Path) -> None:
    write_canonical_json(
        path,
        {
            "campaign_seed": 20260901,
            "noise_grid": [0.0375],
            "selection_mode": "fixed",
            "pilot_shots_per_point": 1,
            "train_shots_cap": 8,
            "calibration_shots_cap": 8,
            "calibration_decode_shots_cap": 8,
            "calibration_shortlist_per_method": 1,
            "test_batch_shots": 1,
            "max_test_shots_per_point": 8,
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


def _write_selection(
    path: Path,
    *,
    config: Path,
    code_manifest: Path,
    selected_noise_points: list[float] | None = None,
    selection_mode: str = "fixed",
) -> None:
    evidence_role = {
        "fixed": "predeclared_selection_not_evidence",
        "pilot": "selection_only_not_held_out",
    }[selection_mode]
    write_canonical_json(
        path,
        {
            "pilot_rows": [],
            "selected_noise_points": (
                [0.003, 0.005] if selected_noise_points is None else selected_noise_points
            ),
            "selection_mode": selection_mode,
            "evidence_role": evidence_role,
            "source_sha256": {
                "code_manifest": sha256_file(code_manifest),
                "config": sha256_file(config),
            },
        },
    )
    write_canonical_json(
        path.parent / "manifest.json",
        {
            "complete": True,
            "role": "pilot",
            "selection_sha256": sha256_file(path),
            "shards": {},
        },
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
    assert selection["selection_mode"] == "pilot"
    assert selection["evidence_role"] == "selection_only_not_held_out"
    assert len(selection["pilot_rows"]) > 1
    assert selection["pilot_rows"][1]["error_rate"] == 0.0045
    pilot_manifest = json.loads((pilot_dir / "manifest.json").read_text())
    assert pilot_manifest["complete"] is True

    missing_pilot_shard = min(pilot_manifest["shards"])
    missing_pilot_shard_path = pilot_dir / missing_pilot_shard
    missing_pilot_shard_bytes = missing_pilot_shard_path.read_bytes()
    missing_pilot_shard_path.unlink()
    rejected_missing_pilot_shard = _run_shard_cli(
        config=config_path,
        code=code_dir,
        selection=selection_path,
        output=campaign_dir / "missing-pilot-shard/train",
        shots=16,
    )
    assert rejected_missing_pilot_shard.returncode != 0
    assert "completion manifest shard set does not match" in rejected_missing_pilot_shard.stderr
    missing_pilot_shard_path.write_bytes(missing_pilot_shard_bytes)

    selection_bytes = selection_path.read_bytes()
    pilot_manifest_path = pilot_dir / "manifest.json"
    pilot_manifest_bytes = pilot_manifest_path.read_bytes()
    rehashed_selection = dict(selection)
    rehashed_selection["selected_noise_points"] = [selection["pilot_rows"][0]["error_rate"]]
    write_canonical_json(selection_path, rehashed_selection)
    pilot_manifest["selection_sha256"] = sha256_file(selection_path)
    write_canonical_json(pilot_manifest_path, pilot_manifest)
    rejected_rehashed_selection = _run_shard_cli(
        config=config_path,
        code=code_dir,
        selection=selection_path,
        output=campaign_dir / "rehashed-selection/train",
        shots=16,
    )
    assert rejected_rehashed_selection.returncode != 0
    assert "selected noise points disagree with pilot rows" in rejected_rehashed_selection.stderr
    selection_path.write_bytes(selection_bytes)
    pilot_manifest_path.write_bytes(pilot_manifest_bytes)

    coordinated_selection = json.loads(selection_bytes)
    first_row = coordinated_selection["pilot_rows"][0]
    first_row["block_errors"] = (
        0 if first_row["block_errors"] else first_row["shots"]
    )
    coordinated_selection["selected_noise_points"] = list(
        select_noise_points(coordinated_selection["pilot_rows"])
    )
    write_canonical_json(selection_path, coordinated_selection)
    coordinated_manifest = json.loads(pilot_manifest_bytes)
    coordinated_manifest["selection_sha256"] = sha256_file(selection_path)
    write_canonical_json(pilot_manifest_path, coordinated_manifest)
    rejected_coordinated_selection = _run_shard_cli(
        config=config_path,
        code=code_dir,
        selection=selection_path,
        output=campaign_dir / "coordinated-selection/train",
        shots=16,
    )
    assert rejected_coordinated_selection.returncode != 0
    assert "block_errors disagree with verified pilot outcomes" in (
        rejected_coordinated_selection.stderr
    )
    selection_path.write_bytes(selection_bytes)
    pilot_manifest_path.write_bytes(pilot_manifest_bytes)

    pilot_row_tampering: tuple[
        tuple[str, object, str],
        ...,
    ] = (
        ("unexpected_field", True, "pilot row schema"),
        (
            "converged",
            0 if selection["pilot_rows"][0]["converged"] else 1,
            "deterministic fields",
        ),
        (
            "latency_mean_seconds",
            float(selection["pilot_rows"][0]["latency_seconds"]) + 1.0,
            "timing fields",
        ),
    )
    for field, value, message in pilot_row_tampering:
        tampered_selection = json.loads(selection_bytes)
        tampered_selection["pilot_rows"][0][field] = value
        write_canonical_json(selection_path, tampered_selection)
        tampered_manifest = json.loads(pilot_manifest_bytes)
        tampered_manifest["selection_sha256"] = sha256_file(selection_path)
        write_canonical_json(pilot_manifest_path, tampered_manifest)
        rejected_row = _run_shard_cli(
            config=config_path,
            code=code_dir,
            selection=selection_path,
            output=campaign_dir / f"tampered-{field}/train",
            shots=16,
        )
        assert rejected_row.returncode != 0
        assert message in rejected_row.stderr
        selection_path.write_bytes(selection_bytes)
        pilot_manifest_path.write_bytes(pilot_manifest_bytes)

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


def test_fixed_selection_cli_publishes_predeclared_rate_without_sampling(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    config_path = tmp_path / "campaign.json"
    pilot_dir = tmp_path / "campaign" / "pilot"
    _write_fixed_config(config_path)
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

    selection = json.loads((pilot_dir / "selection.json").read_text())
    manifest = json.loads((pilot_dir / "manifest.json").read_text())
    assert selection["selection_mode"] == "fixed"
    assert selection["selected_noise_points"] == [0.0375]
    assert selection["pilot_rows"] == []
    assert selection["evidence_role"] == "predeclared_selection_not_evidence"
    assert manifest["shards"] == {}
    assert not list(pilot_dir.glob("rate-*"))


def test_fixed_selection_cli_rejects_tampering_after_publication(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    config_path = tmp_path / "campaign.json"
    pilot_dir = tmp_path / "campaign" / "pilot"
    _write_fixed_config(config_path)
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
    selection["selected_noise_points"] = [0.008]
    selection_path.write_text(json.dumps(selection))

    result = _run_shard_cli(
        config=config_path,
        code=code_dir,
        selection=selection_path,
        output=tmp_path / "train",
    )

    assert result.returncode != 0
    assert "selection SHA-256 mismatch" in result.stderr


def test_fixed_selection_cli_rejects_selection_mode_mismatch(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    config_path = tmp_path / "campaign.json"
    pilot_dir = tmp_path / "pilot"
    pilot_dir.mkdir()
    selection_path = pilot_dir / "selection.json"
    _write_noncanonical_code(code_dir)
    _write_fixed_config(config_path)
    _write_selection(
        selection_path,
        config=config_path,
        code_manifest=code_dir / "code.json",
        selection_mode="pilot",
    )

    result = _run_shard_cli(
        config=config_path,
        code=code_dir,
        selection=selection_path,
        output=tmp_path / "train",
    )

    assert result.returncode != 0
    assert "selection mode does not match campaign configuration" in result.stderr


def test_fixed_selection_cli_rejects_rate_mismatch(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    config_path = tmp_path / "campaign.json"
    pilot_dir = tmp_path / "pilot"
    pilot_dir.mkdir()
    selection_path = pilot_dir / "selection.json"
    _write_noncanonical_code(code_dir)
    _write_fixed_config(config_path)
    _write_selection(
        selection_path,
        config=config_path,
        code_manifest=code_dir / "code.json",
        selected_noise_points=[0.008],
        selection_mode="fixed",
    )

    result = _run_shard_cli(
        config=config_path,
        code=code_dir,
        selection=selection_path,
        output=tmp_path / "train",
    )

    assert result.returncode != 0
    assert "fixed selection rates do not match configured noise_grid" in result.stderr


def test_pilot_cli_rejects_noncanonical_code(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    config_path = tmp_path / "campaign.json"
    _write_noncanonical_code(code_dir)
    _write_reduced_config(config_path, selection_mode="pilot")

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
