from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str, *, label: str) -> None:
    """Fail if an artifact no longer matches its manifest digest."""
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, found {actual}")


def write_canonical_json(path: Path, value: Mapping[str, object]) -> None:
    """Write stable, human-readable JSON for a scientific artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_manifest(
    *,
    command: list[str],
    inputs: list[Path],
    outputs: list[Path],
    parameters: Mapping[str, object],
) -> dict[str, object]:
    """Describe a run and content-address all of its declared artifacts."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "command": command,
        "git_commit": commit,
        "inputs": {str(path): sha256_file(path) for path in inputs},
        "outputs": {str(path): sha256_file(path) for path in outputs},
        "parameters": dict(parameters),
        "platform": platform.platform(),
        "python": sys.version,
    }
