from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from qldpc_fno.artifacts import sha256_file

CLI_PATH = Path("experiments/34_audit_planar_diagnostics.py")


def _cli_module():
    spec = importlib.util.spec_from_file_location("adaptive_diagnostics_cli", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _action(probabilities: list[float], flops: int) -> dict[str, object]:
    return {
        "probabilities": probabilities,
        "selected_class": int(np.argmax(probabilities)),
        "work": {
            "estimated_arithmetic_flops": flops,
            "truncation_events": [
                {
                    "spectral_summaries": [
                        {
                            "discarded_squared_weight_fraction": 0.01,
                            "spectral_entropy": 0.2,
                            "retained_rank": 2,
                        }
                    ]
                }
            ],
        },
    }


def _shot(shot_id: str, *, mismatch: bool) -> dict[str, object]:
    return {
        "shot_id": shot_id,
        "shot_index": int(shot_id.rsplit("i", 1)[1]),
        "error_rate": 0.1,
        "accepted": True,
        "used_fallback": False,
        "exact_class": 1,
        "policy_class": 0 if mismatch else 1,
        "class_mismatch": mismatch,
        "outcome_discordance": mismatch,
        "reference_valid": True,
        "reference_certificate": {"probability_margins": [0.2, 0.2]},
        "tensor": {
            "exact_columns": _action([0.3, 0.5, 0.1, 0.1], 30),
            "exact_rows": _action([0.3, 0.5, 0.1, 0.1], 30),
            "tolerance_columns": _action(
                [0.6, 0.2, 0.1, 0.1] if mismatch else [0.2, 0.6, 0.1, 0.1], 10
            ),
            "tolerance_rows": _action(
                [0.55, 0.25, 0.1, 0.1] if mismatch else [0.25, 0.55, 0.1, 0.1], 10
            ),
            "fixed_columns": _action([0.31, 0.49, 0.1, 0.1], 20),
            "fallback_columns": None,
        },
        "arms": {
            "exact": {"failure": mismatch},
            "policy": {"failure": False},
        },
        "work": {"policy": 20, "fixed_chi8": 20, "exact": 60},
        "invalid_tensor_work": {},
    }


def _write_source_and_manifest(tmp_path: Path) -> tuple[Path, Path]:
    artifact = tmp_path / "source.json"
    source = {
        "schema_version": 1,
        "status": "falsified_exact_outcome_preservation",
        "shots": [
            _shot("d5/p0.100000/i000000", mismatch=True),
            _shot("d5/p0.100000/i000001", mismatch=False),
        ],
        "per_rate": [
            {
                "error_rate": 0.1,
                "shots": 2,
                "class_mismatches": 1,
                "outcome_discordances": 1,
                "fallbacks": 0,
            }
        ],
        "work": {"totals": {"policy": 40, "fixed_chi8": 40, "exact": 120}},
    }
    artifact.write_text(json.dumps(source, sort_keys=True))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_binding": {
                    "raw": {
                        "sha256": sha256_file(artifact),
                        "size_bytes": artifact.stat().st_size,
                    }
                }
            }
        )
    )
    return artifact, manifest


def test_cli_writes_a_source_bound_diagnostic_audit_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    module = _cli_module()
    artifact, manifest = _write_source_and_manifest(tmp_path)
    output_dir = tmp_path / "result"

    result = module.run(
        ["--artifact", str(artifact), "--manifest", str(manifest), "--out", str(output_dir)]
    )

    output_path = output_dir / "diagnostic_audit.json"
    assert output_path.is_file()
    assert json.loads(output_path.read_text()) == result
    assert result["source"]["sha256"] == sha256_file(artifact)
    assert result["source"]["size_bytes"] == artifact.stat().st_size
    assert result["source"]["publication_manifest_sha256"] == sha256_file(manifest)
    assert result["source_consistency"]["passed"] is True
    with pytest.raises(FileExistsError):
        module.run(
            [
                "--artifact",
                str(artifact),
                "--manifest",
                str(manifest),
                "--out",
                str(output_dir),
            ]
        )


def test_cli_rejects_source_size_or_hash_mismatch(tmp_path: Path) -> None:
    module = _cli_module()
    artifact, manifest = _write_source_and_manifest(tmp_path)
    manifest_payload = json.loads(manifest.read_text())
    manifest_payload["artifact_binding"]["raw"]["size_bytes"] += 1
    manifest.write_text(json.dumps(manifest_payload))

    with pytest.raises(ValueError, match="size mismatch"):
        module.run(
            [
                "--artifact",
                str(artifact),
                "--manifest",
                str(manifest),
                "--out",
                str(tmp_path / "result"),
            ]
        )


def test_cli_rejects_same_size_sha_mismatch_without_writing_output(tmp_path: Path) -> None:
    module = _cli_module()
    artifact, manifest = _write_source_and_manifest(tmp_path)
    original = artifact.read_bytes()
    artifact.write_bytes(b"[" + original[1:])
    output_dir = tmp_path / "result"

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.run(
            [
                "--artifact",
                str(artifact),
                "--manifest",
                str(manifest),
                "--out",
                str(output_dir),
            ]
        )

    assert not output_dir.exists()
