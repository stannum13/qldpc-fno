from __future__ import annotations

import argparse
from pathlib import Path

import ldpc
import stim

from qldpc_fno.artifacts import write_canonical_json

EXPECTED_STIM_VERSION = "1.16.0"
EXPECTED_LDPC_VERSION = "2.4.1"


def source_lock() -> dict[str, object]:
    """Return the primary-source registry used by the smoke experiment."""
    if stim.__version__ != EXPECTED_STIM_VERSION:
        raise RuntimeError(f"expected Stim {EXPECTED_STIM_VERSION}, found {stim.__version__}")
    if ldpc.__version__ != EXPECTED_LDPC_VERSION:
        raise RuntimeError(f"expected ldpc {EXPECTED_LDPC_VERSION}, found {ldpc.__version__}")
    return {
        "paper": {
            "version": "2603.28627v1",
            "url": "https://arxiv.org/abs/2603.28627v1",
        },
        "stim": {
            "version": stim.__version__,
            "url": "https://github.com/quantumlib/Stim/tree/v1.16.0",
        },
        "ldpc": {
            "version": ldpc.__version__,
            "url": "https://pypi.org/project/ldpc/2.4.1/",
        },
        "willow": {
            "doi": "10.5281/zenodo.13273331",
            "license": "CC-BY-4.0",
            "url": "https://zenodo.org/records/13273331",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_canonical_json(args.out, source_lock())


if __name__ == "__main__":
    main()
