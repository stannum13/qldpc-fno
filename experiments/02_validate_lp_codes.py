from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy import sparse

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.codes.lifted_product import CSSCode, validate_css


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads((args.code / "code.json").read_text())
    hx_path = args.code / "hx.npz"
    hz_path = args.code / "hz.npz"
    if sha256_file(hx_path) != metadata["hx_sha256"]:
        raise ValueError("hx.npz does not match code.json")
    if sha256_file(hz_path) != metadata["hz_sha256"]:
        raise ValueError("hz.npz does not match code.json")

    code = CSSCode(
        name=metadata["name"],
        ell=metadata["ell"],
        hx=sparse.load_npz(hx_path).tocsr(),
        hz=sparse.load_npz(hz_path).tocsr(),
        n=metadata["n"],
        k=metadata["k"],
        distance_upper_bound=metadata["distance_upper_bound"],
    )
    checks = validate_css(code)
    matches_paper = (
        checks["hx_shape"] == [945, 2610]
        and checks["hz_shape"] == [945, 2610]
        and checks["n"] == 2610
        and checks["k"] == 744
        and checks["hx_row_weights"] == {"min": 10, "max": 10}
        and checks["hz_row_weights"] == {"min": 10, "max": 10}
    )
    checks["matches_paper"] = matches_paper
    checks["valid"] = bool(checks["valid"] and matches_paper)
    write_canonical_json(args.code / "checks.json", checks)
    if not checks["valid"]:
        raise SystemExit("code validation failed; inspect checks.json")


if __name__ == "__main__":
    main()
