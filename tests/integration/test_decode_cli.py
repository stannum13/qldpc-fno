import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

from qldpc_fno.artifacts import sha256_file
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
    hx_path = code_dir / "hx.npz"
    logical_path = dem_dir / "logical_x.npz"
    detections_path = samples_dir / "dets.b8"
    observables_path = samples_dir / "obs_actual.b8"
    sparse.save_npz(hx_path, hx)
    sparse.save_npz(logical_path, logical_x)
    write_b8(detections_path, np.array([[1, 0], [0, 1]], dtype=np.uint8))
    write_b8(observables_path, np.array([[1], [1]], dtype=np.uint8))
    code_manifest_path = code_dir / "code.json"
    code_manifest_path.write_text(json.dumps({"hx_sha256": sha256_file(hx_path)}))
    (dem_dir / "dem.json").write_text(
        json.dumps(
            {
                "code_metadata_sha256": sha256_file(code_manifest_path),
                "error_rate": 0.05,
                "logical_x_sha256": sha256_file(logical_path),
                "model_sha256": "dem-source",
            }
        )
    )
    (samples_dir / "samples.json").write_text(
        json.dumps(
            {
                "shots": 2,
                "num_detectors": 2,
                "num_observables": 1,
                "source_dem_sha256": "dem-source",
                "sha256": {
                    "detections": sha256_file(detections_path),
                    "observables_actual": sha256_file(observables_path),
                },
            }
        )
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
