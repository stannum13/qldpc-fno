from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

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
            "noise_grid": [0.2, 0.25],
            "selection_mode": "pilot",
            "pilot_shots_per_point": 1,
            "train_shots_cap": 4,
            "calibration_shots_cap": 2,
            "calibration_decode_shots_cap": 2,
            "calibration_shortlist_per_method": 1,
            "test_batch_shots": 1,
            "max_test_shots_per_point": 2,
            "target_failures": 1,
            "test_stopping_mode": "fixed",
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
            "evidence_role": "selection_only_not_held_out",
            "pilot_rows": [],
            "selected_noise_points": [0.2, 0.25],
            "selection_mode": "pilot",
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
        1,
        "--out",
        out,
    )
    assert result.returncode == 0, result.stderr


def _evaluate(
    *,
    config: Path,
    code: Path,
    selection: Path,
    test: Path,
    model: Path,
    calibration: Path,
    out: Path,
    extra: tuple[object, ...] = (),
) -> subprocess.CompletedProcess[str]:
    return _run(
        "experiments/17_evaluate_hybrid_decoders.py",
        "--config",
        config,
        "--code",
        code,
        "--selection",
        selection,
        "--test",
        test,
        "--model",
        model,
        "--calibration",
        calibration,
        "--campaign-mode",
        "reduced_non_scientific",
        "--out",
        out,
        *extra,
    )


def test_fixed_paired_evaluation_is_atomic_resumable_and_provenance_strict(
    tmp_path: Path,
) -> None:
    config = tmp_path / "campaign.json"
    code = tmp_path / "code"
    campaign = tmp_path / "campaign"
    pilot = campaign / "pilot"
    train = campaign / "train"
    calibration = campaign / "calibration"
    test = campaign / "test"
    model = campaign / "model"
    evaluation = campaign / "evaluation"
    pilot.mkdir(parents=True)
    selection = pilot / "selection.json"
    _write_config(config)

    built = _run("experiments/01_build_lp_codes.py", "--out", code)
    assert built.returncode == 0, built.stderr
    _write_selection(selection, config=config, code_manifest=code / "code.json")
    _generate_role(config=config, code=code, selection=selection, role="train", shots=4, out=train)
    _generate_role(
        config=config,
        code=code,
        selection=selection,
        role="calibration",
        shots=2,
        out=calibration,
    )
    _generate_role(config=config, code=code, selection=selection, role="test", shots=2, out=test)

    trained = _run(
        "experiments/15_train_conditional_fno.py",
        "--config",
        config,
        "--code",
        code,
        "--train",
        train,
        "--out",
        model,
    )
    assert trained.returncode == 0, trained.stderr
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
        1,
        "--campaign-mode",
        "reduced_non_scientific",
        "--out",
        calibration,
    )
    assert calibrated.returncode == 0, calibrated.stderr

    selected_path = calibration / "selected.json"
    selected_text = selected_path.read_text()
    canonical_reject = _run(
        "experiments/17_evaluate_hybrid_decoders.py",
        "--config",
        config,
        "--code",
        code,
        "--selection",
        selection,
        "--test",
        test,
        "--model",
        model,
        "--calibration",
        calibration,
        "--out",
        evaluation,
    )
    assert canonical_reject.returncode != 0
    assert "two-stage policy" in canonical_reject.stderr

    grid_path = calibration / "grid.json"
    grid_text = grid_path.read_text()
    impossible_proxy = json.loads(grid_text)
    impossible_proxy["screening_candidates"][0][
        "mean_residual_syndrome_weight"
    ] = 10_000.0
    write_canonical_json(grid_path, impossible_proxy)
    selected = json.loads(selected_text)
    selected["source_sha256"]["grid"] = sha256_file(grid_path)
    write_canonical_json(selected_path, selected)
    impossible_proxy_reject = _evaluate(
        config=config,
        code=code,
        selection=selection,
        test=test,
        model=model,
        calibration=calibration,
        out=evaluation,
    )
    assert impossible_proxy_reject.returncode != 0
    assert "screening grid is inconsistent" in impossible_proxy_reject.stderr
    grid_path.write_text(grid_text)
    selected_path.write_text(selected_text)

    incomplete_grid = json.loads(grid_text)
    incomplete_grid["screening_candidates"].pop()
    write_canonical_json(grid_path, incomplete_grid)
    selected = json.loads(selected_text)
    selected["source_sha256"]["grid"] = sha256_file(grid_path)
    write_canonical_json(selected_path, selected)
    incomplete_reject = _evaluate(
        config=config,
        code=code,
        selection=selection,
        test=test,
        model=model,
        calibration=calibration,
        out=evaluation,
    )
    assert incomplete_reject.returncode != 0
    assert "two-stage policy is malformed" in incomplete_reject.stderr
    grid_path.write_text(grid_text)
    selected_path.write_text(selected_text)

    selected = json.loads(selected_text)
    selected["source_sha256"]["config"] = "0" * 64
    write_canonical_json(selected_path, selected)
    rejected_calibration = _evaluate(
        config=config,
        code=code,
        selection=selection,
        test=test,
        model=model,
        calibration=calibration,
        out=evaluation,
    )
    assert rejected_calibration.returncode != 0
    assert "calibration configuration provenance" in rejected_calibration.stderr
    selected_path.write_text(selected_text)

    model_path = model / "model.pt"
    model_bytes = model_path.read_bytes()
    model_path.write_bytes(model_bytes + b"corrupt")
    rejected_model = _evaluate(
        config=config,
        code=code,
        selection=selection,
        test=test,
        model=model,
        calibration=calibration,
        out=evaluation,
    )
    assert rejected_model.returncode != 0
    assert "model SHA-256 mismatch" in rejected_model.stderr
    model_path.write_bytes(model_bytes)

    test_completion_path = test / "manifest.json"
    test_completion_text = test_completion_path.read_text()
    test_completion = json.loads(test_completion_text)
    first_relative = min(test_completion["shards"])
    test_shard_path = test / first_relative
    test_shard_text = test_shard_path.read_text()
    test_shard = json.loads(test_shard_text)
    test_shard["seed"] += 1
    write_canonical_json(test_shard_path, test_shard)
    test_completion["shards"][first_relative] = sha256_file(test_shard_path)
    write_canonical_json(test_completion_path, test_completion)
    rejected_test = _evaluate(
        config=config,
        code=code,
        selection=selection,
        test=test,
        model=model,
        calibration=calibration,
        out=evaluation,
    )
    assert rejected_test.returncode != 0
    assert "derived seed" in rejected_test.stderr
    test_shard_path.write_text(test_shard_text)
    test_completion_path.write_text(test_completion_text)

    partial = _evaluate(
        config=config,
        code=code,
        selection=selection,
        test=test,
        model=model,
        calibration=calibration,
        out=evaluation,
        extra=("--max-batches-this-run", 1),
    )
    assert partial.returncode == 0, partial.stderr
    assert json.loads((evaluation / "progress.json").read_text())["status"] == "in_progress"
    partial_manifests = sorted(evaluation.glob("rate-*/batch-*/manifest.json"))
    assert len(partial_manifests) == 1
    first_batch_manifest = partial_manifests[0]
    first_batch = json.loads(first_batch_manifest.read_text())
    assert first_batch["failures"] == {"baseline": 1, "residual": 1, "soft_prior": 1}
    first_outcomes = first_batch_manifest.parent / first_batch["outcomes_path"]
    first_bytes = first_outcomes.read_bytes()
    first_stat = first_outcomes.stat()
    assert first_batch["outcomes_sha256"] == sha256_file(first_outcomes)
    assert not list(evaluation.rglob("*.tmp"))

    laundered_deadline = campaign / "evaluation-laundered-deadline"
    shutil.copytree(evaluation, laundered_deadline)
    finalized_under_cap = _evaluate(
        config=config,
        code=code,
        selection=selection,
        test=test,
        model=model,
        calibration=calibration,
        out=laundered_deadline,
        extra=("--resume", "--deadline-monotonic", 0),
    )
    assert finalized_under_cap.returncode == 0, finalized_under_cap.stderr
    laundered_manifest_path = laundered_deadline / "manifest.json"
    laundered_manifest = json.loads(laundered_manifest_path.read_text())
    assert laundered_manifest["status"] == "partial_deadline"
    assert laundered_manifest["rates"]["0"]["status"] == "partial_deadline"
    assert laundered_manifest["rates"]["0"]["shots"] == 1
    laundered_manifest["status"] = "complete"
    write_canonical_json(laundered_manifest_path, laundered_manifest)
    laundered_batches = {
        path: path.read_bytes()
        for path in sorted(laundered_deadline.glob("rate-*/batch-*/manifest.json"))
    }
    rejected_laundering = _evaluate(
        config=config,
        code=code,
        selection=selection,
        test=test,
        model=model,
        calibration=calibration,
        out=laundered_deadline,
        extra=("--resume",),
    )
    assert rejected_laundering.returncode != 0
    assert "complete evaluation requires every rate to be complete" in (
        rejected_laundering.stderr
    )
    assert sorted(laundered_deadline.glob("rate-*/batch-*/manifest.json")) == list(
        laundered_batches
    )
    assert all(path.read_bytes() == payload for path, payload in laundered_batches.items())

    immediate_deadline = campaign / "evaluation-immediate-deadline"
    deadline = _evaluate(
        config=config,
        code=code,
        selection=selection,
        test=test,
        model=model,
        calibration=calibration,
        out=immediate_deadline,
        extra=("--deadline-monotonic", 0),
    )
    assert deadline.returncode == 0, deadline.stderr
    deadline_manifest = json.loads((immediate_deadline / "manifest.json").read_text())
    assert deadline_manifest["complete"] is True
    assert deadline_manifest["status"] == "partial_deadline"
    assert deadline_manifest["rates"]["0"]["shots"] == 0
    assert deadline_manifest["rates"]["1"]["shots"] == 0
    partial_summary = json.loads(
        (immediate_deadline / deadline_manifest["rates"]["0"]["summary_path"]).read_text()
    )
    assert partial_summary["comparison_status"]["soft_prior"] == "not_fixed_sample"
    assert partial_summary["comparison_status"]["residual"] == "not_fixed_sample"
    assert "accuracy_compatible" not in partial_summary

    resumed_deadline = _evaluate(
        config=config,
        code=code,
        selection=selection,
        test=test,
        model=model,
        calibration=calibration,
        out=immediate_deadline,
        extra=("--resume",),
    )
    assert resumed_deadline.returncode == 0, resumed_deadline.stderr
    assert json.loads((immediate_deadline / "manifest.json").read_text())["status"] == "complete"

    tampered_evaluation = campaign / "evaluation-tampered"
    shutil.copytree(evaluation, tampered_evaluation)
    tampered_manifest_path = next(tampered_evaluation.glob("rate-*/batch-*/manifest.json"))
    tampered_manifest = json.loads(tampered_manifest_path.read_text())
    tampered_outcomes_path = tampered_manifest_path.parent / tampered_manifest["outcomes_path"]
    with np.load(tampered_outcomes_path, allow_pickle=False) as archive:
        tampered_arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    tampered_arrays["baseline_failure"] ^= True
    with tampered_outcomes_path.open("wb") as handle:
        np.savez_compressed(handle, **tampered_arrays)
    tampered_manifest["outcomes_sha256"] = sha256_file(tampered_outcomes_path)
    write_canonical_json(tampered_manifest_path, tampered_manifest)
    tampered_progress_path = tampered_evaluation / "progress.json"
    tampered_progress = json.loads(tampered_progress_path.read_text())
    relative_manifest = str(tampered_manifest_path.relative_to(tampered_evaluation))
    tampered_progress["rates"]["0"]["batch_manifests"][relative_manifest] = sha256_file(
        tampered_manifest_path
    )
    write_canonical_json(tampered_progress_path, tampered_progress)
    rejected_semantics = _evaluate(
        config=config,
        code=code,
        selection=selection,
        test=test,
        model=model,
        calibration=calibration,
        out=tampered_evaluation,
        extra=("--resume",),
    )
    assert rejected_semantics.returncode != 0
    assert "failure outcomes" in rejected_semantics.stderr

    tampered_reliability = campaign / "evaluation-tampered-reliability"
    shutil.copytree(evaluation, tampered_reliability)
    reliability_manifest_path = next(tampered_reliability.glob("rate-*/batch-*/manifest.json"))
    reliability_manifest = json.loads(reliability_manifest_path.read_text())
    rows = reliability_manifest["probability_reliability"]["soft_prior"]
    populated_index = next(index for index, row in enumerate(rows) if row["count"] > 0)
    populated = rows[populated_index]
    populated["probability_sum"] = float(populated["count"]) if populated_index < 9 else 0.0
    write_canonical_json(reliability_manifest_path, reliability_manifest)
    reliability_progress_path = tampered_reliability / "progress.json"
    reliability_progress = json.loads(reliability_progress_path.read_text())
    relative_reliability_manifest = str(reliability_manifest_path.relative_to(tampered_reliability))
    reliability_progress["rates"]["0"]["batch_manifests"][relative_reliability_manifest] = (
        sha256_file(reliability_manifest_path)
    )
    write_canonical_json(reliability_progress_path, reliability_progress)
    rejected_reliability = _evaluate(
        config=config,
        code=code,
        selection=selection,
        test=test,
        model=model,
        calibration=calibration,
        out=tampered_reliability,
        extra=("--resume",),
    )
    assert rejected_reliability.returncode != 0
    assert "reliability probability sum" in rejected_reliability.stderr

    resumed = _evaluate(
        config=config,
        code=code,
        selection=selection,
        test=test,
        model=model,
        calibration=calibration,
        out=evaluation,
        extra=("--resume",),
    )
    assert resumed.returncode == 0, resumed.stderr
    assert first_outcomes.read_bytes() == first_bytes
    assert first_outcomes.stat().st_ino == first_stat.st_ino
    assert first_outcomes.stat().st_mtime_ns == first_stat.st_mtime_ns

    manifest = json.loads((evaluation / "manifest.json").read_text())
    assert manifest["complete"] is True
    assert manifest["status"] == "complete"
    assert manifest["selection_mode"] == "pilot"
    assert manifest["test_stopping_mode"] == "fixed"
    assert manifest["target_failures_active"] is False
    assert set(manifest["source_sha256"]) == {
        "calibration_grid",
        "calibration_manifest",
        "calibration_selected",
        "calibration_shard_manifests",
        "code_manifest",
        "config",
        "model_manifest",
        "selection",
        "test_manifest",
        "test_shard_manifests",
    }
    assert set(manifest["rates"]) == {"0", "1"}
    selected = json.loads(selected_path.read_text())
    for method in ("soft_prior", "residual"):
        assert manifest["selected_calibration"][method] == {
            "model_checkpoint": selected["selected"][method]["model_checkpoint"],
            "model_epoch": selected["selected"][method]["model_epoch"],
            "parameters": selected["selected"][method]["parameters"],
        }

    batch_manifests = sorted(evaluation.glob("rate-*/batch-*/manifest.json"))
    assert len(batch_manifests) == 4
    for batch_manifest_path in batch_manifests:
        batch_manifest = json.loads(batch_manifest_path.read_text())
        outcomes_path = batch_manifest_path.parent / batch_manifest["outcomes_path"]
        assert batch_manifest["outcomes_sha256"] == sha256_file(outcomes_path)
        with np.load(outcomes_path, allow_pickle=False) as outcomes:
            assert outcomes["shot_indices"].shape == (1,)
            indices_sha256 = hashlib.sha256(
                outcomes["shot_indices"].astype("<i8", copy=False).tobytes()
            ).hexdigest()
            assert batch_manifest["shot_indices_sha256"] == indices_sha256
            for method in ("soft_prior", "residual"):
                reliability = batch_manifest["probability_reliability"][method]
                assert len(reliability) == 10
                assert sum(row["count"] for row in reliability) == 2_610
            for decoder in ("baseline", "soft_prior", "residual"):
                failure = outcomes[f"{decoder}_failure"]
                invalid = ~outcomes[f"{decoder}_syndrome_valid"]
                mismatch = outcomes[f"{decoder}_logical_mismatch"]
                assert np.array_equal(failure, invalid | mismatch)
            for method in ("soft_prior", "residual"):
                assert np.allclose(
                    outcomes[f"{method}_end_to_end_latency_seconds"],
                    outcomes[f"{method}_latency_seconds"]
                    + outcomes[f"{method}_fno_latency_seconds"],
                )

    for rate_record in manifest["rates"].values():
        summary_path = evaluation / str(rate_record["summary_path"])
        assert rate_record["summary_sha256"] == sha256_file(summary_path)
        summary = json.loads(summary_path.read_text())
        assert summary["status"] == "complete"
        assert summary["stop_reason"] == "shot_cap"
        assert summary["shots"] == 2
        assert "accuracy_compatible" not in summary
        for method in ("soft_prior", "residual"):
            reliability = summary["diagnostics"][f"{method}_probability_reliability"]
            assert len(reliability) == 10
            assert sum(row["count"] for row in reliability) == 2 * 2_610
            assert summary["diagnostics"][f"{method}_end_to_end_latency_seconds"]["mean"] >= 0
        for method in ("soft_prior", "residual"):
            paired = summary["paired"][method]
            assert summary["comparison_status"][method] in {
                "harm_detected",
                "benefit_detected",
                "inconclusive",
                "no_discordances",
            }
            assert "block_error_delta_95ci_low" not in paired
            assert "block_error_delta_95ci_high" not in paired

    completed_bytes = {path: path.read_bytes() for path in batch_manifests}
    verified_resume = _evaluate(
        config=config,
        code=code,
        selection=selection,
        test=test,
        model=model,
        calibration=calibration,
        out=evaluation,
        extra=("--resume",),
    )
    assert verified_resume.returncode == 0, verified_resume.stderr
    assert all(path.read_bytes() == payload for path, payload in completed_bytes.items())

    missing_exact = campaign / "evaluation-missing-exact"
    shutil.copytree(evaluation, missing_exact)
    missing_manifest_path = missing_exact / "manifest.json"
    missing_manifest = json.loads(missing_manifest_path.read_text())
    missing_summary_path = missing_exact / missing_manifest["rates"]["0"]["summary_path"]
    missing_summary = json.loads(missing_summary_path.read_text())
    missing_summary["paired"]["soft_prior"].pop("mcnemar_exact_pvalue_harm")
    write_canonical_json(missing_summary_path, missing_summary)
    missing_manifest["rates"]["0"]["summary_sha256"] = sha256_file(missing_summary_path)
    write_canonical_json(missing_manifest_path, missing_manifest)
    missing_batches = {
        path: path.read_bytes() for path in sorted(missing_exact.glob("rate-*/batch-*/manifest.json"))
    }
    rejected_missing_exact = _evaluate(
        config=config,
        code=code,
        selection=selection,
        test=test,
        model=model,
        calibration=calibration,
        out=missing_exact,
        extra=("--resume",),
    )
    assert rejected_missing_exact.returncode != 0
    assert "exact paired fields" in rejected_missing_exact.stderr
    assert all(path.read_bytes() == payload for path, payload in missing_batches.items())

    under_cap = campaign / "evaluation-under-cap"
    shutil.copytree(evaluation, under_cap)
    under_cap_manifest_path = under_cap / "manifest.json"
    under_cap_manifest = json.loads(under_cap_manifest_path.read_text())
    under_cap_summary_path = under_cap / under_cap_manifest["rates"]["0"]["summary_path"]
    under_cap_summary = json.loads(under_cap_summary_path.read_text())
    under_cap_summary["shots"] = 1
    write_canonical_json(under_cap_summary_path, under_cap_summary)
    under_cap_manifest["rates"]["0"]["shots"] = 1
    under_cap_manifest["rates"]["0"]["summary_sha256"] = sha256_file(under_cap_summary_path)
    write_canonical_json(under_cap_manifest_path, under_cap_manifest)
    under_cap_batches = {
        path: path.read_bytes() for path in sorted(under_cap.glob("rate-*/batch-*/manifest.json"))
    }
    rejected_under_cap = _evaluate(
        config=config,
        code=code,
        selection=selection,
        test=test,
        model=model,
        calibration=calibration,
        out=under_cap,
        extra=("--resume",),
    )
    assert rejected_under_cap.returncode != 0
    assert "fixed evaluation rate must complete at the configured shot cap" in (
        rejected_under_cap.stderr
    )
    assert all(path.read_bytes() == payload for path, payload in under_cap_batches.items())

    manifest_path = evaluation / "manifest.json"
    manifest_text = manifest_path.read_text()
    malformed_manifest = json.loads(manifest_text)
    malformed_manifest["rates"]["99"] = malformed_manifest["rates"]["0"]
    write_canonical_json(manifest_path, malformed_manifest)
    rejected_final = _evaluate(
        config=config,
        code=code,
        selection=selection,
        test=test,
        model=model,
        calibration=calibration,
        out=evaluation,
        extra=("--resume",),
    )
    assert rejected_final.returncode != 0
    assert "rate coordinates" in rejected_final.stderr
    manifest_path.write_text(manifest_text)
