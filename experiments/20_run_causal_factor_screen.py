from __future__ import annotations

import argparse
from pathlib import Path

from qldpc_fno.temporal.screen import run_reduced_screen, verify_screen_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or verify the reduced causal factor screen")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--sequences", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--config", type=Path, required=True)
    verify.add_argument("--sequences", type=Path, required=True)
    verify.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run_reduced_screen(
            config_path=args.config,
            sequence_dir=args.sequences,
            output_dir=args.out,
        )
    else:
        verify_screen_result(args.out, config_path=args.config, sequence_dir=args.sequences)


if __name__ == "__main__":
    main()
