"""Run the local-spectrum tolerance truncation discovery study."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from qldpc_fno.decision.tensor_tolerance_study import run_tensor_tolerance_study


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def run(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    return run_tensor_tolerance_study(args.config, args.reference, args.out)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
