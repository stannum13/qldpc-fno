from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.campaign.shard_io import load_verified_shards


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
            "noise_grid": [0.003, 0.005],
            "pilot_shots_per_point": 1,
            "train_shots_cap": 16,
            "calibration_shots_cap": 8,
            "calibration_decode_shots_cap": 8,
            "calibration_shortlist_per_method": 1,
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
        4,
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

    train = campaign / "train"
    train_completion_path = train / "manifest.json"
    train_completion_text = train_completion_path.read_text()
    train_completion = json.loads(train_completion_text)
    manifest_relatives = sorted(train_completion["shards"])
    first_manifest_path = train / manifest_relatives[0]
    first_manifest_text = first_manifest_path.read_text()
    first_manifest = json.loads(first_manifest_text)

    first_manifest["shots"] = float(first_manifest["shots"])
    write_canonical_json(first_manifest_path, first_manifest)
    train_completion["shards"][manifest_relatives[0]] = sha256_file(first_manifest_path)
    write_canonical_json(train_completion_path, train_completion)
    with pytest.raises(ValueError, match="shots must be a positive integer"):
        load_verified_shards(
            train,
            role="train",
            config_path=config,
            code_manifest_path=code / "code.json",
        )
    first_manifest_path.write_text(first_manifest_text)
    train_completion_path.write_text(train_completion_text)

    first_manifest = json.loads(first_manifest_text)
    first_manifest["dimensions"]["dets.b8"] = True
    write_canonical_json(first_manifest_path, first_manifest)
    train_completion = json.loads(train_completion_text)
    train_completion["shards"][manifest_relatives[0]] = sha256_file(first_manifest_path)
    write_canonical_json(train_completion_path, train_completion)
    with pytest.raises(ValueError, match="dimensions must contain exact non-negative integers"):
        load_verified_shards(
            train,
            role="train",
            config_path=config,
            code_manifest_path=code / "code.json",
        )
    first_manifest_path.write_text(first_manifest_text)
    train_completion_path.write_text(train_completion_text)

    first_manifest = json.loads(first_manifest_text)
    first_manifest["sha256"].pop("obs_actual.b8")
    write_canonical_json(first_manifest_path, first_manifest)
    train_completion["shards"][manifest_relatives[0]] = sha256_file(first_manifest_path)
    write_canonical_json(train_completion_path, train_completion)
    with pytest.raises(ValueError, match="exactly dets.b8, errors.b8, and obs_actual.b8"):
        load_verified_shards(
            train,
            role="train",
            config_path=config,
            code_manifest_path=code / "code.json",
        )
    first_manifest_path.write_text(first_manifest_text)
    train_completion_path.write_text(train_completion_text)

    first_manifest = json.loads(first_manifest_text)
    first_manifest["seed"] += 1
    write_canonical_json(first_manifest_path, first_manifest)
    train_completion = json.loads(train_completion_text)
    train_completion["shards"][manifest_relatives[0]] = sha256_file(first_manifest_path)
    write_canonical_json(train_completion_path, train_completion)
    with pytest.raises(ValueError, match="derived seed"):
        load_verified_shards(
            train,
            role="train",
            config_path=config,
            code_manifest_path=code / "code.json",
        )
    first_manifest_path.write_text(first_manifest_text)
    train_completion_path.write_text(train_completion_text)

    second_manifest_path = train / manifest_relatives[1]
    second_manifest_text = second_manifest_path.read_text()
    second_manifest = json.loads(second_manifest_text)
    second_manifest["rate_index"] = first_manifest["rate_index"]
    second_manifest["shard_index"] = first_manifest["shard_index"]
    second_manifest["seed"] = first_manifest["seed"]
    write_canonical_json(second_manifest_path, second_manifest)
    train_completion = json.loads(train_completion_text)
    train_completion["shards"][manifest_relatives[1]] = sha256_file(second_manifest_path)
    write_canonical_json(train_completion_path, train_completion)
    with pytest.raises(ValueError, match="duplicate shard coordinate"):
        load_verified_shards(
            train,
            role="train",
            config_path=config,
            code_manifest_path=code / "code.json",
        )
    second_manifest_path.write_text(second_manifest_text)
    train_completion_path.write_text(train_completion_text)

    initialized = _run(
        "experiments/15_train_conditional_fno.py",
        "--config",
        config,
        "--code",
        code,
        "--train",
        train,
        "--initialize-only",
        "--out",
        model,
    )
    assert initialized.returncode == 0, initialized.stderr
    initialization = json.loads((model / "resume.json").read_text())
    assert initialization["status"] == "initialized"
    assert initialization["checkpoint"] is None
    assert initialization["teacher_metadata_sha256"] is None
    assert not (model / "teacher.json").exists()
    assert not (model / "model.json").exists()

    teacher_partial = _run(
        "experiments/15_train_conditional_fno.py",
        "--config",
        config,
        "--code",
        code,
        "--train",
        train,
        "--resume",
        "--max-teacher-chunks-this-run",
        1,
        "--out",
        model,
    )
    assert teacher_partial.returncode == 0, teacher_partial.stderr
    teacher_partial_resume = json.loads((model / "resume.json").read_text())
    assert teacher_partial_resume["status"] == "teacher_partial"
    assert not (model / "teacher.json").exists()
    assert not list(model.glob("epoch-*.pt"))
    teacher_chunk_manifests = sorted((model / "teacher_chunks").glob("chunk-*.json"))
    assert len(teacher_chunk_manifests) == 1
    first_chunk_manifest = json.loads(teacher_chunk_manifests[0].read_text())
    first_chunk_path = model / "teacher_chunks" / first_chunk_manifest["path"]
    first_chunk_bytes = first_chunk_path.read_bytes()
    first_chunk_stat = first_chunk_path.stat()

    partial = _run(
        "experiments/15_train_conditional_fno.py",
        "--config",
        config,
        "--code",
        code,
        "--train",
        train,
        "--resume",
        "--max-epochs-this-run",
        1,
        "--out",
        model,
    )
    assert partial.returncode == 0, partial.stderr
    assert first_chunk_path.read_bytes() == first_chunk_bytes
    assert first_chunk_path.stat().st_ino == first_chunk_stat.st_ino
    assert first_chunk_path.stat().st_mtime_ns == first_chunk_stat.st_mtime_ns
    assert not (model / "model.json").exists()
    resume_metadata = json.loads((model / "resume.json").read_text())
    assert resume_metadata["status"] == "checkpointed"
    teacher_path = model / "teacher.json"
    original_teacher = teacher_path.read_text()
    teacher_metadata = json.loads(original_teacher)
    teacher_metadata["positive_counts_by_channel"][0] += 1
    write_canonical_json(teacher_path, teacher_metadata)

    rejected_teacher = _run(
        "experiments/15_train_conditional_fno.py",
        "--config",
        config,
        "--code",
        code,
        "--train",
        train,
        "--resume",
        "--out",
        model,
    )
    assert rejected_teacher.returncode != 0
    assert "teacher metadata SHA-256 mismatch" in rejected_teacher.stderr
    teacher_path.write_text(original_teacher)

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
        train,
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
        train,
        "--resume",
        "--out",
        model,
    )
    assert resumed.returncode == 0, resumed.stderr
    model_metadata = json.loads((model / "model.json").read_text())
    completed_resume = json.loads((model / "resume.json").read_text())
    assert completed_resume["status"] == "complete"
    assert completed_resume["model_manifest_sha256"] == sha256_file(model / "model.json")
    assert model_metadata["complete"] is True
    assert model_metadata["source_role"] == "train"
    assert set(model_metadata["source_sha256"]) == {
        "code_manifest",
        "config",
        "train_manifest",
        "train_shard_manifests",
    }
    assert len(model_metadata["source_sha256"]["train_shard_manifests"]) == 4
    assert sha256_file(model / "model.pt") == model_metadata["sha256"]
    assert model_metadata["teacher"]["decoder"]["lsd_order"] == 5
    assert model_metadata["teacher"]["metadata_sha256"] == sha256_file(teacher_path)
    assert len(model_metadata["checkpoints"]) == 2
    assert {candidate["epoch"] for candidate in model_metadata["checkpoints"]} == {1, 2}
    per_rate = model_metadata["split"]["per_rate"]
    assert {row["error_rate"] for row in per_rate} == {0.003, 0.005}
    assert all(row["train_shots"] > 0 and row["validation_shots"] > 0 for row in per_rate)
    assert sum(row["train_shots"] for row in per_rate) == model_metadata["split"]["train_shots"]
    assert (
        sum(row["validation_shots"] for row in per_rate)
        == model_metadata["split"]["validation_shots"]
    )

    completed_teacher = json.loads(teacher_path.read_text())
    completed_chunk_manifest_path = model / min(completed_teacher["chunks"])
    completed_chunk_manifest = json.loads(completed_chunk_manifest_path.read_text())
    completed_chunk_path = completed_chunk_manifest_path.parent / completed_chunk_manifest["path"]
    completed_chunk = completed_chunk_path.read_bytes()
    completed_chunk_path.unlink()
    rejected_missing_chunk = _run(
        "experiments/15_train_conditional_fno.py",
        "--config",
        config,
        "--code",
        code,
        "--train",
        train,
        "--resume",
        "--out",
        model,
    )
    assert rejected_missing_chunk.returncode != 0
    assert "teacher chunk" in rejected_missing_chunk.stderr
    completed_chunk_path.write_bytes(completed_chunk)

    corrupted_chunk = bytearray(completed_chunk)
    corrupted_chunk[0] ^= 1
    completed_chunk_path.write_bytes(corrupted_chunk)
    rejected_corrupted_chunk = _run(
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
        "--campaign-mode",
        "reduced_non_scientific",
        "--out",
        calibration,
    )
    assert rejected_corrupted_chunk.returncode != 0
    assert "teacher chunk" in rejected_corrupted_chunk.stderr
    completed_chunk_path.write_bytes(completed_chunk)

    teacher_corrections_path = model / "teacher_corrections.b8"
    teacher_corrections = teacher_corrections_path.read_bytes()
    teacher_corrections_path.unlink()
    rejected_missing_teacher = _run(
        "experiments/15_train_conditional_fno.py",
        "--config",
        config,
        "--code",
        code,
        "--train",
        train,
        "--resume",
        "--out",
        model,
    )
    assert rejected_missing_teacher.returncode != 0
    assert "teacher correction cache" in rejected_missing_teacher.stderr
    teacher_corrections_path.write_bytes(teacher_corrections)

    canonical_reduced_control = _run(
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
    assert canonical_reduced_control.returncode != 0
    assert "canonical calibration cannot reduce" in canonical_reduced_control.stderr

    corrupted_teacher = bytearray(teacher_corrections)
    corrupted_teacher[0] ^= 1
    teacher_corrections_path.write_bytes(corrupted_teacher)
    rejected_corrupted_teacher = _run(
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
        "--campaign-mode",
        "reduced_non_scientific",
        "--out",
        calibration,
    )
    assert rejected_corrupted_teacher.returncode != 0
    assert "teacher correction cache SHA-256 mismatch" in rejected_corrupted_teacher.stderr
    teacher_corrections_path.write_bytes(teacher_corrections)

    partial_calibration = _run(
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
        "--campaign-mode",
        "reduced_non_scientific",
        "--max-work-units-this-run",
        1,
        "--out",
        calibration,
    )
    assert partial_calibration.returncode == 0, partial_calibration.stderr
    progress = json.loads((calibration / "progress.json").read_text())
    assert progress["completed_work_units"] == 1
    assert progress["total_work_units"] == 4
    assert progress["shortlists"] is None
    assert len(progress["screening_candidates"]) == 2
    assert progress["hybrid_candidates"] == []
    assert not (calibration / "selected.json").exists()

    progress_path = calibration / "progress.json"
    partial_progress_text = progress_path.read_text()
    corrupted_progress = json.loads(partial_progress_text)
    corrupted_progress["screening_candidates"][0]["logits_sha256"] = "not-a-sha256"
    write_canonical_json(progress_path, corrupted_progress)
    try:
        rejected_screening_progress = _run(
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
            "--campaign-mode",
            "reduced_non_scientific",
            "--max-work-units-this-run",
            1,
            "--resume",
            "--out",
            calibration,
        )
    finally:
        progress_path.write_text(partial_progress_text)
    assert rejected_screening_progress.returncode != 0
    assert "screening order or measurement" in rejected_screening_progress.stderr

    second_partial = _run(
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
        "--campaign-mode",
        "reduced_non_scientific",
        "--max-work-units-this-run",
        1,
        "--resume",
        "--out",
        calibration,
    )
    assert second_partial.returncode == 0, second_partial.stderr
    transition_progress = json.loads(progress_path.read_text())
    assert transition_progress["hybrid_candidates"] == []
    assert transition_progress["shortlists"] is not None
    transition_progress["shortlists"] = None
    transition_progress["decode_work_indices"] = None
    write_canonical_json(progress_path, transition_progress)
    recovered_transition = _run(
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
        "--campaign-mode",
        "reduced_non_scientific",
        "--max-work-units-this-run",
        1,
        "--resume",
        "--out",
        calibration,
    )
    assert recovered_transition.returncode == 0, recovered_transition.stderr
    second_progress_text = progress_path.read_text()
    second_progress = json.loads(second_progress_text)
    assert len(second_progress["hybrid_candidates"]) == 1
    second_progress["hybrid_candidates"][0]["residual"]["invalid_count"] = 9
    write_canonical_json(progress_path, second_progress)
    try:
        rejected_hybrid_progress = _run(
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
            "--campaign-mode",
            "reduced_non_scientific",
            "--max-work-units-this-run",
            1,
            "--resume",
            "--out",
            calibration,
        )
    finally:
        progress_path.write_text(second_progress_text)
    assert rejected_hybrid_progress.returncode != 0
    assert "hybrid residual score" in rejected_hybrid_progress.stderr

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
        "--campaign-mode",
        "reduced_non_scientific",
        "--resume",
        "--out",
        calibration,
    )
    assert calibrated.returncode == 0, calibrated.stderr
    grid = json.loads((calibration / "grid.json").read_text())
    selected = json.loads((calibration / "selected.json").read_text())
    assert len(grid["screening_candidates"]) == 4
    assert 1 <= len(grid["hybrid_candidates"]) <= 2
    assert set(grid["shortlists"]) == {"residual", "soft_prior"}
    assert all(len(indices) == 1 for indices in grid["shortlists"].values())
    assert grid["policy"]["decode_subset"]["shots"] == 8
    assert grid["policy"]["decode_subset"]["indices_sha256"]
    assert {row["model_epoch"] for row in grid["screening_candidates"]} == {1, 2}
    assert all(
        set(row)
        >= {
            "inference_latency_seconds",
            "logits_sha256",
            "model_checkpoint",
            "model_epoch",
            "parameters",
            "residual",
            "soft_prior",
            "work_index",
        }
        for row in grid["hybrid_candidates"]
    )
    for epoch in {row["model_epoch"] for row in grid["screening_candidates"]}:
        epoch_rows = [
            row for row in grid["screening_candidates"] if row["model_epoch"] == epoch
        ]
        assert len({row["logits_sha256"] for row in epoch_rows}) == 1
        assert len({row["inference_latency_seconds"] for row in epoch_rows}) == 1
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
    assert len(selected["source_sha256"]["calibration_shard_manifests"]) == 2
    assert selected["selection_rule"] == [
        "has_any_invalid_correction",
        "block_errors",
        "nll",
        "model_epoch",
        "alpha",
        "beta",
        "temperature",
    ]
    for method in ("soft_prior", "residual"):
        selected_checkpoint = selected["selected"][method]["model_checkpoint"]
        assert sha256_file(model / selected_checkpoint["path"]) == selected_checkpoint["sha256"]
        assert selected["selected"][method] in [
            row[method]
            | {
                "inference_latency_seconds": row["inference_latency_seconds"],
                "model_checkpoint": row["model_checkpoint"],
                    "model_epoch": row["model_epoch"],
                    "parameters": row["parameters"],
                    "screening_proxy": row["screening_proxy"],
                    "work_index": row["work_index"],
                }
            for row in grid["hybrid_candidates"]
            if row["work_index"] in grid["shortlists"][method]
        ]
