"""Calibrate one frozen CMWPM decoder configuration from planar calibration shots."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from qldpc_fno.decision.planar_shot_accuracy import calibrate_cmwpm


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def run(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    return calibrate_cmwpm(args.grid, args.data, args.out)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
