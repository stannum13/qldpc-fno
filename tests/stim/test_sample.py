import json
from pathlib import Path

import stim

from qldpc_fno.stim.b8 import read_b8
from qldpc_fno.stim.sample import sample_dem_shard


def test_sample_manifest_and_files(tmp_path: Path) -> None:
    dem = stim.DetectorErrorModel("error(0.25) D0 L0")
    manifest = sample_dem_shard(dem, shots=32, seed=123, output_dir=tmp_path)
    assert manifest["shots"] == 32
    assert manifest["seed"] == 123
    assert read_b8(tmp_path / "dets.b8", shots=32, bits_per_shot=1).shape == (32, 1)
    assert read_b8(tmp_path / "obs_actual.b8", shots=32, bits_per_shot=1).shape == (32, 1)
    assert read_b8(tmp_path / "errors.b8", shots=32, bits_per_shot=1).shape == (32, 1)
    assert json.loads((tmp_path / "samples.json").read_text()) == manifest
