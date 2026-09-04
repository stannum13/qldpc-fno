from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.campaign.evaluation import (
    _read_verified_json,
    _VerifiedBatch,
    _verify_batch_trajectory,
    _verify_existing_progress,
    _verify_orphan_rate_summaries,
)


def _record(
    *,
    start: int,
    shots: int,
    failures: dict[str, int],
) -> _VerifiedBatch:
    return _VerifiedBatch(
        manifest_path=Path(f"batch-{start:05d}/manifest.json"),
        manifest={
            "failures": failures,
            "invalid": {"baseline": 0, "soft_prior": 0, "residual": 0},
            "logical_mismatch": {"baseline": 0, "soft_prior": 0, "residual": 0},
            "shots": shots,
            "start": start,
            "stop": start + shots,
        },
        manifest_sha256="",
        expected_indices=np.arange(start, start + shots, dtype=np.int64),
    )


def test_verified_batch_trajectory_rejects_oversized_batch() -> None:
    config = SimpleNamespace(
        max_test_shots_per_point=4,
        target_failures=3,
        test_batch_shots=1,
        test_stopping_mode="fixed",
    )
    records = [
        _record(
            start=0,
            shots=2,
            failures={"baseline": 0, "soft_prior": 0, "residual": 0},
        )
    ]

    with pytest.raises(ValueError, match="configured batch size"):
        _verify_batch_trajectory(records, config=config)  # type: ignore[arg-type]


def test_verified_batch_trajectory_rejects_batch_after_adaptive_stop() -> None:
    config = SimpleNamespace(
        max_test_shots_per_point=4,
        target_failures=1,
        test_batch_shots=1,
        test_stopping_mode="adaptive",
    )
    records = [
        _record(
            start=0,
            shots=1,
            failures={"baseline": 1, "soft_prior": 1, "residual": 1},
        ),
        _record(
            start=1,
            shots=1,
            failures={"baseline": 0, "soft_prior": 0, "residual": 0},
        ),
    ]

    with pytest.raises(ValueError, match="after the configured stopping rule"):
        _verify_batch_trajectory(records, config=config)  # type: ignore[arg-type]


def test_operational_progress_recovers_one_atomic_trailing_batch(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    first_path = output / "rate-000/batch-00000/manifest.json"
    second_path = output / "rate-000/batch-00001/manifest.json"
    first_path.parent.mkdir(parents=True)
    second_path.parent.mkdir(parents=True)
    first_path.write_text('{"batch": 0}\n')
    second_path.write_text('{"batch": 1}\n')
    failures = {"baseline": 0, "soft_prior": 0, "residual": 0}
    first = _record(start=0, shots=1, failures=failures)
    second = _record(start=1, shots=1, failures=failures)
    records = [
        _VerifiedBatch(
            first_path,
            first.manifest,
            sha256_file(first_path),
            first.expected_indices,
        ),
        _VerifiedBatch(
            second_path,
            second.manifest,
            sha256_file(second_path),
            second.expected_indices,
        ),
    ]
    source = {"config": "a" * 64}
    write_canonical_json(
        output / "progress.json",
        {
            "rates": {
                "0": {
                    "batch_manifests": {
                        "rate-000/batch-00000/manifest.json": sha256_file(first_path)
                    },
                    "error_rate": 0.2,
                    "failures": failures,
                    "invalid": failures,
                    "logical_mismatch": failures,
                    "shots": 1,
                }
            },
            "source_sha256": source,
            "status": "in_progress",
        },
    )

    assert (
        _verify_existing_progress(
            output,
            source_sha256=source,
            rates=(0.2,),
            rate_records={0: records},
            strict=False,
        )
        == "in_progress"
    )
    with pytest.raises(ValueError, match="batch-manifest provenance mismatch"):
        _verify_existing_progress(
            output,
            source_sha256=source,
            rates=(0.2,),
            rate_records={0: records},
            strict=True,
        )


def test_verified_json_hashes_exact_bytes_instead_of_normalized_newlines(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "summary.json"
    artifact.write_bytes(b'{"value": 1}\n')
    expected_sha256 = sha256_file(artifact)
    artifact.write_bytes(b'{"value": 1}\r\n')

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _read_verified_json(artifact, expected_sha256, label="rate summary")


def test_terminal_progress_requires_every_orphan_summary(tmp_path: Path) -> None:
    config = SimpleNamespace(
        max_test_shots_per_point=1,
        target_failures=2,
        test_batch_shots=1,
        test_stopping_mode="fixed",
    )
    records = {
        0: [
            _record(
                start=0,
                shots=1,
                failures={"baseline": 0, "soft_prior": 0, "residual": 0},
            )
        ]
    }

    with pytest.raises(ValueError, match="requires the complete orphan summary set"):
        _verify_orphan_rate_summaries(
            tmp_path,
            source_sha256={},
            rates=(0.2,),
            records=records,
            config=config,  # type: ignore[arg-type]
            progress_status="complete",
        )
