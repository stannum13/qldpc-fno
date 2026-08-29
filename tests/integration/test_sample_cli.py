import json
import subprocess
import sys
from pathlib import Path

import stim


def test_sample_code_capacity_cli(tmp_path: Path) -> None:
    model_path = tmp_path / "model.dem"
    output_dir = tmp_path / "samples"
    stim.DetectorErrorModel("error(0.25) D0 L0").to_file(model_path)
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
