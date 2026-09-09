"""Train and evaluate the exact-space trajectory-balance GFlowNet."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from qldpc_fno.decision.gflownet_study import run_gflownet_study


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--decision-config", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def run(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    return run_gflownet_study(args.config, args.decision_config, args.baseline, args.out)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
