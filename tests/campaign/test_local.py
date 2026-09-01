from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from qldpc_fno.campaign.local import verify_canonical_checkout


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True)


def test_canonical_checkout_rejects_dirty_config_and_source(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    config = repo / "configs/accuracy_campaign.json"
    source = repo / "src/qldpc_fno/example.py"
    config.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    config.write_text('{"canonical": true}\n')
    source.write_text("VALUE = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "initial",
    )

    commit = verify_canonical_checkout(repo, config)

    assert len(commit) == 40

    config.write_text('{"canonical": false}\n')
    with pytest.raises(ValueError, match="differs from HEAD"):
        verify_canonical_checkout(repo, config)

    _git(repo, "restore", "configs/accuracy_campaign.json")
    source.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="working tree must be clean"):
        verify_canonical_checkout(repo, config)


def test_canonical_checkout_rejects_untracked_execution_inputs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    config = repo / "configs/accuracy_campaign.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "initial",
    )
    (repo / "untracked.py").write_text("print('dirty')\n")

    with pytest.raises(ValueError, match="working tree must be clean"):
        verify_canonical_checkout(repo, config)
