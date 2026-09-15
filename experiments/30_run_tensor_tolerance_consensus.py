"""Evaluate the frozen two-view tolerance consensus policy."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from qldpc_fno.decision.tensor_tolerance_consensus import (
    run_tensor_tolerance_consensus_study,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def run(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    return run_tensor_tolerance_consensus_study(args.policy, args.data, args.out)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
