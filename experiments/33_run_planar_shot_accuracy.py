"""Evaluate and independently replay a frozen planar shot accuracy screen."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from qldpc_fno.decision.planar_shot_accuracy import run_planar_shot_accuracy


def run(argv: Sequence[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return run_planar_shot_accuracy(args.policy, args.screen, args.selection, args.out)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
