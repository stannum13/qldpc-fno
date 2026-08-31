import json
import os
import subprocess
import sys
from pathlib import Path


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def test_reduced_one_command_smoke_loop(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    environment = os.environ.copy()
    environment.update(
        {
            "SMOKE_OUTPUT": str(output),
            "SMOKE_SHOTS": "16",
            "SMOKE_STEPS": "2",
        }
    )
    subprocess.run(["bash", "scripts/run_smoke.sh"], check=True, env=environment)

    assert (output / "source-lock.json").exists()
    checks = _read_json(output / "code" / "checks.json")
    dem = _read_json(output / "dem" / "dem.json")
    samples = _read_json(output / "samples" / "samples.json")
    teacher = _read_json(output / "bplsd" / "decode.json")
    split = _read_json(output / "tensors" / "split.json")
    training = _read_json(output / "fno" / "train_metrics.json")
    evaluation = _read_json(output / "evaluation" / "metrics.json")
    assert checks["valid"] is True
    assert dem["num_detectors"] == 945
    assert dem["num_observables"] == 744
    assert samples["shots"] == 16
    assert teacher["syndrome_valid"] == 16
    assert split["syndrome_shape"] == [16, 21, 45]
    assert split["correction_shape"] == [16, 58, 45]
    assert training["shots"] == 12
    assert evaluation["shots"] == 4

    model_manifest_path = output / "fno" / "model.json"
    original_model_manifest = model_manifest_path.read_text()
    model_manifest = json.loads(original_model_manifest)
    model_manifest["source_sha256"]["syndromes"] = "0" * 64
    model_manifest_path.write_text(json.dumps(model_manifest))
    mixed_result = subprocess.run(
        [
            sys.executable,
            "experiments/12_evaluate_in_size.py",
            "--code",
            str(output / "code"),
            "--dem",
            str(output / "dem"),
            "--samples",
            str(output / "samples"),
            "--tensors",
            str(output / "tensors"),
            "--model",
            str(output / "fno" / "model.pt"),
            "--out",
            str(output / "evaluation-mixed"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert mixed_result.returncode != 0
    assert "source hashes do not match" in mixed_result.stderr
    model_manifest_path.write_text(original_model_manifest)

    subprocess.run(
        [
            sys.executable,
            "experiments/06_sample_code_capacity.py",
            "--dem",
            str(output / "dem" / "model.dem"),
            "--shots",
            "16",
            "--seed",
            "20260830",
            "--out",
            str(output / "samples-other"),
        ],
        check=True,
    )
    cross_run_result = subprocess.run(
        [
            sys.executable,
            "experiments/12_evaluate_in_size.py",
            "--code",
            str(output / "code"),
            "--dem",
            str(output / "dem"),
            "--samples",
            str(output / "samples-other"),
            "--tensors",
            str(output / "tensors"),
            "--model",
            str(output / "fno" / "model.pt"),
            "--out",
            str(output / "evaluation-cross-run"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert cross_run_result.returncode != 0
    assert "samples manifest does not match tensor source" in cross_run_result.stderr

    syndrome_path = output / "tensors" / "syndromes.npy"
    syndrome_path.write_bytes(syndrome_path.read_bytes() + b"corruption")
    corrupted_result = subprocess.run(
        [
            sys.executable,
            "experiments/10_overfit_tiny_models.py",
            "--tensors",
            str(output / "tensors"),
            "--code",
            str(output / "code"),
            "--shots",
            "1",
            "--steps",
            "1",
            "--seed",
            "17",
            "--out",
            str(output / "fno-corrupted"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert corrupted_result.returncode != 0
    assert "syndrome tensor SHA-256 mismatch" in corrupted_result.stderr
