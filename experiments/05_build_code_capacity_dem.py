from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy import sparse

from qldpc_fno.artifacts import sha256_file, verify_sha256, write_canonical_json
from qldpc_fno.codes.gf2 import logical_x_basis
from qldpc_fno.stim.dem import build_z_error_dem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--error-rate", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    code_metadata_path = args.code / "code.json"
    code_metadata = json.loads(code_metadata_path.read_text())
    hx_path = args.code / "hx.npz"
    hz_path = args.code / "hz.npz"
    verify_sha256(hx_path, code_metadata["hx_sha256"], label="Hx")
    verify_sha256(hz_path, code_metadata["hz_sha256"], label="Hz")
    hx = sparse.load_npz(hx_path).tocsr()
    hz = sparse.load_npz(hz_path).tocsr()
    logical_x = logical_x_basis(hx, hz)
    dem = build_z_error_dem(hx, logical_x, error_rate=args.error_rate)

    logical_path = args.out / "logical_x.npz"
    model_path = args.out / "model.dem"
    sparse.save_npz(logical_path, logical_x)
    dem.to_file(model_path)
    write_canonical_json(
        args.out / "dem.json",
        {
            "code": code_metadata["name"],
            "code_metadata_sha256": sha256_file(code_metadata_path),
            "error_rate": args.error_rate,
            "logical_x_sha256": sha256_file(logical_path),
            "model_sha256": sha256_file(model_path),
            "num_detectors": dem.num_detectors,
            "num_errors": dem.num_errors,
            "num_observables": dem.num_observables,
            "noise_model": "independent_z_error",
            "stim_version": "1.16.0",
        },
    )


if __name__ == "__main__":
    main()
