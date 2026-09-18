"""Generate immutable physical-error shots for the planar accuracy screen."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from qldpc_fno.decision.planar_shot_data import generate_planar_shots


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def run(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    return generate_planar_shots(args.config, args.out)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
