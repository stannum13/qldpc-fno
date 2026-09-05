from __future__ import annotations

import argparse
from pathlib import Path

from qldpc_fno.temporal.screen import generate_sequence_campaign, verify_sequence_campaign


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or verify reduced causal sequences")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--out", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--config", type=Path, required=True)
    verify.add_argument("--out", type=Path, required=True)
    verify.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    if args.command == "generate":
        generate_sequence_campaign(config_path=args.config, output_dir=args.out)
    else:
        verify_sequence_campaign(
            config_path=args.config, output_dir=args.out, regenerate=args.regenerate
        )


if __name__ == "__main__":
    main()
