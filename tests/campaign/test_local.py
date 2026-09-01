from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from qldpc_fno.campaign.local import resolve_git_commit, verify_canonical_checkout


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


def test_canonical_checkout_ignores_container_commit_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setenv("CAMPAIGN_GIT_COMMIT", "0123456789abcdef0123456789abcdef01234567")

    assert verify_canonical_checkout(repo, config) == actual


def test_commit_resolver_uses_validated_image_binding_without_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setenv("CAMPAIGN_GIT_COMMIT", commit)

    assert resolve_git_commit(tmp_path / "not-a-repository") == commit


def test_commit_resolver_rejects_invalid_image_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAMPAIGN_GIT_COMMIT", "HEAD; touch /tmp/not-a-commit")

    with pytest.raises(ValueError, match="CAMPAIGN_GIT_COMMIT"):
        resolve_git_commit(tmp_path / "not-a-repository")
