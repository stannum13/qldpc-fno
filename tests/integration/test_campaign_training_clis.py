from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from qldpc_fno.artifacts import sha256_file, write_canonical_json


def _run(*arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *(str(argument) for argument in arguments)],
        capture_output=True,
        check=False,
        text=True,
    )


def _write_config(path: Path) -> None:
    write_canonical_json(
        path,
        {
            "campaign_seed": 20260901,
            "noise_grid": [0.003],
            "pilot_shots_per_point": 1,
            "train_shots_cap": 16,
            "calibration_shots_cap": 8,
            "test_batch_shots": 1,
            "max_test_shots_per_point": 1,
            "target_failures": 1,
            "training_epochs": 2,
            "training_batch_size": 4,
            "training_learning_rate": 0.001,
            "training_seed": 1701,
            "checkpoint_every_epochs": 1,
            "cloud_cpu": 1,
            "cloud_memory": "1Gi",
            "cloud_timeout_seconds": 60,
            "checkpoint_grace_seconds": 1,
        },
    )


def _write_selection(path: Path, *, config: Path, code_manifest: Path) -> None:
    write_canonical_json(
        path,
        {
            "pilot_rows": [],
            "selected_noise_points": [0.003],
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


def _generate_role(
    *, config: Path, code: Path, selection: Path, role: str, shots: int, out: Path
) -> None:
    result = _run(
        "experiments/14_generate_campaign_shards.py",
        "--config",
        config,
        "--code",
        code,
        "--selection",
        selection,
        "--role",
        role,
        "--shots-per-rate",
        shots,
        "--shard-size",
        8,
        "--out",
        out,
    )
    assert result.returncode == 0, result.stderr


def test_reduced_training_resume_and_independent_hybrid_calibration(tmp_path: Path) -> None:
    config = tmp_path / "campaign.json"
    code = tmp_path / "code"
    campaign = tmp_path / "campaign"
    pilot = campaign / "pilot"
    pilot.mkdir(parents=True)
    selection = pilot / "selection.json"
    model = campaign / "model"
    calibration = campaign / "calibration"
    _write_config(config)

    build = _run("experiments/01_build_lp_codes.py", "--out", code)
    assert build.returncode == 0, build.stderr
    _write_selection(selection, config=config, code_manifest=code / "code.json")
    _generate_role(
        config=config,
        code=code,
        selection=selection,
        role="train",
        shots=16,
        out=campaign / "train",
    )
    _generate_role(
        config=config,
        code=code,
        selection=selection,
        role="calibration",
        shots=8,
        out=calibration,
    )

    partial = _run(
        "experiments/15_train_conditional_fno.py",
        "--config",
        config,
        "--code",
        code,
        "--train",
        campaign / "train",
        "--max-epochs-this-run",
        1,
        "--out",
        model,
    )
    assert partial.returncode == 0, partial.stderr
    assert not (model / "model.json").exists()
    resume_metadata = json.loads((model / "resume.json").read_text())
    checkpoint = model / str(resume_metadata["checkpoint"])
    original_checkpoint = checkpoint.read_bytes()
    checkpoint.write_bytes(original_checkpoint + b"corrupt")

    rejected = _run(
        "experiments/15_train_conditional_fno.py",
        "--config",
        config,
        "--code",
        code,
        "--train",
        campaign / "train",
        "--resume",
        "--out",
        model,
    )
    assert rejected.returncode != 0
    assert "checkpoint SHA-256 mismatch" in rejected.stderr
    checkpoint.write_bytes(original_checkpoint)

    resumed = _run(
        "experiments/15_train_conditional_fno.py",
        "--config",
        config,
        "--code",
        code,
        "--train",
        campaign / "train",
        "--resume",
        "--out",
        model,
    )
    assert resumed.returncode == 0, resumed.stderr
    model_metadata = json.loads((model / "model.json").read_text())
    assert model_metadata["complete"] is True
    assert model_metadata["source_role"] == "train"
    assert set(model_metadata["source_sha256"]) == {
        "code_manifest",
        "config",
        "train_manifest",
        "train_shard_manifests",
    }
    assert len(model_metadata["source_sha256"]["train_shard_manifests"]) == 2
    assert sha256_file(model / "model.pt") == model_metadata["sha256"]
    assert model_metadata["teacher"]["decoder"]["lsd_order"] == 5

    calibrated = _run(
        "experiments/16_calibrate_hybrid_priors.py",
        "--config",
        config,
        "--code",
        code,
        "--calibration",
        calibration,
        "--model",
        model,
        "--grid-limit",
        2,
        "--out",
        calibration,
    )
    assert calibrated.returncode == 0, calibrated.stderr
    grid = json.loads((calibration / "grid.json").read_text())
    selected = json.loads((calibration / "selected.json").read_text())
    assert len(grid["candidates"]) == 2
    assert all(set(row) >= {"parameters", "residual", "soft_prior"} for row in grid["candidates"])
    assert selected["complete"] is True
    assert selected["source_role"] == "calibration"
    assert set(selected["selected"]) == {"residual", "soft_prior"}
    assert set(selected["source_sha256"]) == {
        "calibration_manifest",
        "calibration_shard_manifests",
        "code_manifest",
        "config",
        "grid",
        "model_manifest",
    }
    assert len(selected["source_sha256"]["calibration_shard_manifests"]) == 1
    for method in ("soft_prior", "residual"):
        assert selected["selected"][method] in [
            row[method] | {"parameters": row["parameters"]} for row in grid["candidates"]
        ]
