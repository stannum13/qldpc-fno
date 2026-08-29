from __future__ import annotations

import argparse
from pathlib import Path

from scipy import sparse

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.codes.lifted_product import build_self_lifted_product
from qldpc_fno.codes.seeds import PAPER_LP_3_7_16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    code = build_self_lifted_product(PAPER_LP_3_7_16)
    hx_path = args.out / "hx.npz"
    hz_path = args.out / "hz.npz"
    sparse.save_npz(hx_path, code.hx)
    sparse.save_npz(hz_path, code.hz)
    write_canonical_json(
        args.out / "code.json",
        {
            "distance_upper_bound": code.distance_upper_bound,
            "ell": code.ell,
            "hx_sha256": sha256_file(hx_path),
            "hz_sha256": sha256_file(hz_path),
            "k": code.k,
            "n": code.n,
            "name": code.name,
            "source": "https://arxiv.org/abs/2603.28627v1",
        },
    )


if __name__ == "__main__":
    main()
