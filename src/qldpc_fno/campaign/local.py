"""Local campaign provenance gates shared by the shell entry point."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _git(repo: Path, *arguments: str, text: bool = False) -> bytes | str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=text,
    ).stdout


def resolve_git_commit(repo: Path) -> str:
    """Resolve provenance from a validated image binding or Git checkout."""
    image_commit = os.environ.get("CAMPAIGN_GIT_COMMIT")
    if image_commit is not None:
        invalid_character = any(
            character not in "0123456789abcdef" for character in image_commit
        )
        if len(image_commit) != 40 or invalid_character:
            raise ValueError("CAMPAIGN_GIT_COMMIT must be a full lowercase SHA-1")
        return image_commit
    commit = _git(repo, "rev-parse", "HEAD", text=True)
    if not isinstance(commit, str) or len(commit.strip()) != 40:
        raise ValueError("Git commit is invalid")
    return commit.strip()


def verify_canonical_checkout(repo: Path, config: Path) -> str:
    """Return ``HEAD`` only when canonical policy and executable checkout are clean."""
    resolved_repo = repo.resolve(strict=True)
    resolved_config = config.resolve(strict=True)
    if config.is_symlink() or not resolved_config.is_file():
        raise ValueError("canonical campaign configuration must be a regular tracked file")
    try:
        relative = resolved_config.relative_to(resolved_repo).as_posix()
    except ValueError as error:
        raise ValueError(
            "canonical campaign configuration must be inside the repository"
        ) from error
    committed = _git(resolved_repo, "show", f"HEAD:{relative}")
    if not isinstance(committed, bytes) or resolved_config.read_bytes() != committed:
        raise ValueError("canonical campaign configuration differs from HEAD")
    status = _git(
        resolved_repo,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    if not isinstance(status, bytes) or status:
        raise ValueError("canonical campaign working tree must be clean")
    commit = _git(resolved_repo, "rev-parse", "HEAD", text=True)
    if not isinstance(commit, str) or len(commit.strip()) != 40:
        raise ValueError("canonical campaign Git commit is invalid")
    return commit.strip()
