"""Generate or independently verify temporal-identifiability sequence artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from qldpc_fno.identifiability.sequence_store import (
    SequenceStoreDependencies,
    generate_campaign,
    verify_campaign,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument(
            "--roles",
            nargs="+",
            required=True,
            help="one or more canonical roles (comma-separated values are also accepted)",
        )
        command.add_argument("--approval", type=Path)
        command.add_argument("--development-record", type=Path)
    return parser


def run(argv: Sequence[str] | None = None, *, dependencies: SequenceStoreDependencies | None = None) -> dict[str, object]:
    """Run the production command; dependency injection is reserved for tests."""
    args = _parser().parse_args(argv)
    parameters = {
        "config_path": args.config,
        "output_dir": args.out,
        "roles": args.roles,
        "approval_path": args.approval,
        "development_record": args.development_record,
        "dependencies": dependencies,
    }
    if args.command == "generate":
        return generate_campaign(**parameters)
    return verify_campaign(**parameters)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
