"""Run or independently verify the temporal-identifiability gate."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from qldpc_fno.identifiability.screen import (
    ScreenDependencies,
    run_confirmation,
    run_development,
    verify_identifiability_run,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("development", "confirmation", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--sequences", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--development-record", type=Path)
        command.add_argument("--approval", type=Path)
    return parser


def run(
    argv: Sequence[str] | None = None,
    *,
    dependencies: ScreenDependencies | None = None,
) -> dict[str, object]:
    args = _parser().parse_args(argv)
    common = {
        "config_path": args.config,
        "sequence_dir": args.sequences,
        "output_dir": args.out,
        "dependencies": dependencies,
    }
    if args.command == "development":
        if args.development_record is not None or args.approval is not None:
            raise ValueError("development does not accept approval inputs")
        return run_development(**common)
    if args.command == "confirmation":
        if args.development_record is None or args.approval is None:
            raise ValueError("confirmation requires development record and approval")
        return run_confirmation(
            **common,
            development_record=args.development_record,
            approval_path=args.approval,
        )
    return verify_identifiability_run(
        **common,
        development_record=args.development_record,
        approval_path=args.approval,
    )


def main() -> None:
    run()


if __name__ == "__main__":
    main()
