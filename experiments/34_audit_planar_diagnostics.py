#!/usr/bin/env python3
"""Derive the development-open planar adaptive-diagnostic audit."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.decision.adaptive_diagnostics import build_diagnostic_audit


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def _raw_binding(manifest_path: Path) -> Mapping[str, object]:
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, Mapping):
        raise TypeError("publication manifest must be a JSON object")
    binding = manifest.get("artifact_binding")
    raw = binding.get("raw") if isinstance(binding, Mapping) else None
    if not isinstance(raw, Mapping):
        raise TypeError("publication manifest is missing artifact_binding.raw")
    if not isinstance(raw.get("sha256"), str) or type(raw.get("size_bytes")) is not int:
        raise ValueError("publication manifest has an invalid raw artifact binding")
    return raw


def run(argv: Sequence[str] | None = None) -> dict[str, object]:
    """Verify the source binding, build the audit, and write canonical JSON."""
    args = _arguments(argv)
    artifact_path = Path(args.artifact)
    output_dir = Path(args.out)
    output_path = output_dir / "diagnostic_audit.json"
    if output_dir.exists() or output_path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic audit output: {output_dir}")
    raw_binding = _raw_binding(Path(args.manifest))
    expected_size = int(raw_binding["size_bytes"])
    actual_size = artifact_path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"source artifact size mismatch: expected {expected_size}, found {actual_size}"
        )
    expected_sha256 = str(raw_binding["sha256"])
    actual_sha256 = sha256_file(artifact_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"source artifact SHA-256 mismatch: expected {expected_sha256}, found {actual_sha256}"
        )
    source = json.loads(artifact_path.read_text())
    if not isinstance(source, Mapping):
        raise TypeError("source artifact must be a JSON object")
    result = build_diagnostic_audit(source, source_sha256=actual_sha256)
    result_source = result["source"]
    assert isinstance(result_source, dict)
    result_source["size_bytes"] = actual_size
    result_source["publication_manifest_sha256"] = sha256_file(Path(args.manifest))
    write_canonical_json(output_path, result)
    return result


if __name__ == "__main__":
    run()
