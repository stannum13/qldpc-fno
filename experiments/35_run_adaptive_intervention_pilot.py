"""Run the frozen adaptive intervention development pilot on replayed physical shots."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from qldpc_fno.decision.adaptive_intervention_pilot import run_adaptive_intervention_pilot


def run(argv: Sequence[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shots", type=Path, required=True)
    parser.add_argument("--shot-config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--non-scientific-fixture",
        action="store_true",
        help="exactly one shot per frozen rate; never scientific or eligible for advancement",
    )
    args = parser.parse_args(argv)
    return run_adaptive_intervention_pilot(
        args.config,
        args.shots,
        args.out,
        shot_config_path=args.shot_config,
        non_scientific_fixture=args.non_scientific_fixture,
    )


def main() -> None:
    run()


if __name__ == "__main__":
    main()
