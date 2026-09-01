from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from qldpc_fno.campaign.runner import CampaignStatus, write_campaign_summary


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--code", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--status",
        choices=tuple(status.value for status in CampaignStatus),
        default=CampaignStatus.COMPLETE.value,
    )
    parser.add_argument("--reason", action="append", default=[])
    parser.add_argument("--git-commit", default=None)
    args = parser.parse_args()
    write_campaign_summary(
        args.campaign,
        args.out,
        completion_state=CampaignStatus(args.status),
        git_commit=args.git_commit or _git_commit(),
        early_stop_reasons=args.reason,
        config_path=args.config,
        code_path=args.code,
    )


if __name__ == "__main__":
    main()
