import json
import subprocess
import sys
from pathlib import Path

import stim

from qldpc_fno.artifacts import sha256_file


def test_sample_code_capacity_cli(tmp_path: Path) -> None:
    model_path = tmp_path / "model.dem"
    output_dir = tmp_path / "samples"
    stim.DetectorErrorModel("error(0.25) D0 L0").to_file(model_path)
    (tmp_path / "dem.json").write_text(json.dumps({"model_sha256": sha256_file(model_path)}))
    subprocess.run(
        [
            sys.executable,
            "experiments/06_sample_code_capacity.py",
            "--dem",
            str(model_path),
            "--shots",
            "16",
            "--seed",
            "123",
            "--out",
            str(output_dir),
        ],
        check=True,
    )
    manifest = json.loads((output_dir / "samples.json").read_text())
    assert manifest["shots"] == 16
    assert manifest["num_detectors"] == 1
    assert manifest["num_observables"] == 1
    assert manifest["num_errors"] == 1


def test_sample_cli_rejects_a_corrupted_dem(tmp_path: Path) -> None:
    model_path = tmp_path / "model.dem"
    stim.DetectorErrorModel("error(0.25) D0 L0").to_file(model_path)
    (tmp_path / "dem.json").write_text(json.dumps({"model_sha256": sha256_file(model_path)}))
    model_path.write_text("error(0.5) D0 L0\n")
    result = subprocess.run(
        [
            sys.executable,
            "experiments/06_sample_code_capacity.py",
            "--dem",
            str(model_path),
            "--shots",
            "1",
            "--seed",
            "123",
            "--out",
            str(tmp_path / "samples"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode != 0
    assert "detector error model SHA-256 mismatch" in result.stderr


def test_sample_cli_rejects_a_dem_directory(tmp_path: Path) -> None:
    dem_directory = tmp_path / "dem"
    dem_directory.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "experiments/06_sample_code_capacity.py",
            "--dem",
            str(dem_directory),
            "--shots",
            "1",
            "--seed",
            "123",
            "--out",
            str(tmp_path / "samples"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode != 0
    assert "DEM file" in result.stderr
