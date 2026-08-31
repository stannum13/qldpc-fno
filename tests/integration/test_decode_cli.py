import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

from qldpc_fno.stim.b8 import read_b8, write_b8


def test_decode_bplsd_cli(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    dem_dir = tmp_path / "dem"
    samples_dir = tmp_path / "samples"
    output_dir = tmp_path / "decoded"
    code_dir.mkdir()
    dem_dir.mkdir()
    samples_dir.mkdir()
    hx = sparse.csr_matrix([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    logical_x = sparse.csr_matrix([[1, 1, 1]], dtype=np.uint8)
    sparse.save_npz(code_dir / "hx.npz", hx)
    sparse.save_npz(dem_dir / "logical_x.npz", logical_x)
    write_b8(samples_dir / "dets.b8", np.array([[1, 0], [0, 1]], dtype=np.uint8))
    write_b8(samples_dir / "obs_actual.b8", np.array([[1], [1]], dtype=np.uint8))
    (samples_dir / "samples.json").write_text(
        json.dumps({"shots": 2, "num_detectors": 2, "num_observables": 1})
    )

    subprocess.run(
        [
            sys.executable,
            "experiments/07_decode_bplsd.py",
            "--code",
            str(code_dir),
            "--dem",
            str(dem_dir),
            "--samples",
            str(samples_dir),
            "--error-rate",
            "0.05",
            "--out",
            str(output_dir),
        ],
        check=True,
    )

    decode = json.loads((output_dir / "decode.json").read_text())
    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert decode["syndrome_valid"] == 2
    assert metrics["block_errors"] == 0
    corrections = read_b8(output_dir / "corrections.b8", shots=2, bits_per_shot=3)
    assert np.array_equal((corrections @ hx.T.toarray()) % 2, [[1, 0], [0, 1]])
