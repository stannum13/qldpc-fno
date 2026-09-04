from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.campaign import evaluation as evaluation_module
from qldpc_fno.campaign.evaluation import (
    _batch_manifest,
    _rate_summary_payload,
    _read_verified_json,
    _retire_partial_finalization,
    _selection_for_evaluation,
    _VerifiedBatch,
    _verify_batch_manifest,
    _verify_batch_trajectory,
    _verify_existing_progress,
    _verify_orphan_rate_summaries,
    _write_outcomes_atomic,
    _write_progress,
)
from qldpc_fno.campaign.runner import CampaignRunner, CampaignStage, StageResult
from qldpc_fno.campaign.storage import LocalArtifactStore


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


def _latency_batch_arrays(shots: int, *, total_latency: float) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name in evaluation_module._BOOLEAN_OUTCOME_FIELDS:
        if name.endswith("_syndrome_valid"):
            arrays[name] = np.ones(shots, dtype=np.bool_)
        else:
            arrays[name] = np.zeros(shots, dtype=np.bool_)
    for name in evaluation_module._INTEGER_OUTCOME_FIELDS:
        arrays[name] = np.zeros(shots, dtype=np.int64)
    arrays["shot_indices"] = np.arange(shots, dtype=np.int64)
    for name in evaluation_module._FLOAT_OUTCOME_FIELDS:
        arrays[name] = np.zeros(shots, dtype=np.float64)
    per_shot = total_latency / shots
    for method in ("soft_prior", "residual"):
        arrays[f"{method}_fno_latency_seconds"][:] = per_shot
        arrays[f"{method}_end_to_end_latency_seconds"][:] = per_shot
    return arrays


@pytest.mark.parametrize("shots", [256, 2_048])
def test_committed_batch_accepts_persisted_latency_rounding_after_atomic_rename(
    tmp_path: Path,
    shots: int,
) -> None:
    total_latency = 8.3
    arrays = _latency_batch_arrays(shots, total_latency=total_latency)
    reliability = {
        method: [
            {
                "bin_high": (bin_index + 1) / 10,
                "bin_low": bin_index / 10,
                "count": shots * 2_610 if bin_index == 0 else 0,
                "observed_errors": 0,
                "probability_sum": 0.0,
            }
            for bin_index in range(10)
        ]
        for method in ("soft_prior", "residual")
    }
    source = {"selection_verification": "a" * 64}
    rate_dir = tmp_path / "evaluation/rate-000"
    staging = rate_dir / ".batch-00000.tmp"
    staging.mkdir(parents=True)
    outcomes_path = staging / "outcomes.npz"
    _write_outcomes_atomic(outcomes_path, arrays)
    manifest = _batch_manifest(
        rate_index=0,
        rate=0.0375,
        batch_index=0,
        start=0,
        arrays=arrays,
        probability_reliability=reliability,
        outcomes_path=outcomes_path,
        source_sha256=source,
    )
    manifest["fno_inference_latency_seconds"] = {
        "soft_prior": total_latency,
        "residual": total_latency,
    }
    write_canonical_json(
        staging / "manifest.json",
        manifest,
    )
    committed = rate_dir / "batch-00000"
    os.replace(staging, committed)

    verified = _verify_batch_manifest(
        committed / "manifest.json",
        rate_index=0,
        rate=0.0375,
        batch_index=0,
        expected_start=0,
        expected_indices=np.arange(shots, dtype=np.int64),
        source_sha256=source,
    )

    assert verified.manifest["fno_inference_latency_seconds"] == {
        "residual": total_latency,
        "soft_prior": total_latency,
    }


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


class _InjectedRetirementCrash(BaseException):
    pass


@pytest.mark.parametrize(
    "crash_after",
    ("manifest", "progress", "summary-2", "summary-1", "summary-0"),
)
def test_partial_retirement_is_resumable_after_every_mutation_and_checkpoint_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_after: str,
) -> None:
    output = tmp_path / "evaluation"
    output.mkdir()
    rates = (0.1, 0.2, 0.3)
    records: dict[int, list[_VerifiedBatch]] = {index: [] for index in range(len(rates))}
    source = {"config": "a" * 64}
    config = SimpleNamespace(
        max_test_shots_per_point=1,
        target_failures=1,
        test_batch_shots=1,
        test_stopping_mode="fixed",
    )
    _write_progress(
        output,
        source_sha256=source,
        rates=rates,
        rate_records=records,
        status="partial_deadline",
    )
    for rate_index, rate in enumerate(rates):
        rate_dir = output / f"rate-{rate_index:03d}"
        rate_dir.mkdir()
        write_canonical_json(
            rate_dir / "summary.json",
            _rate_summary_payload(
                rate_index=rate_index,
                rate=rate,
                records=[],
                status="partial_deadline",
                stop_reason="campaign_deadline",
                fixed_sample=False,
                source_sha256=source,
            ),
        )
    write_canonical_json(output / "manifest.json", {"status": "partial_deadline"})

    store = LocalArtifactStore(tmp_path / "store")
    stage = CampaignStage(
        "evaluation",
        output,
        lambda deadline: StageResult.CHECKPOINTED,
    )
    runner = CampaignRunner(store=store, stages=(stage,), checkpoint_grace_seconds=0)
    runner._publish_checkpoint(stage, 0, None)

    original_unlink = Path.unlink
    original_write_progress = evaluation_module._write_progress
    if crash_after == "progress":

        def write_then_crash(*args: object, **kwargs: object) -> None:
            original_write_progress(*args, **kwargs)  # type: ignore[arg-type]
            raise _InjectedRetirementCrash

        monkeypatch.setattr(evaluation_module, "_write_progress", write_then_crash)
    else:
        target = (
            output / "manifest.json"
            if crash_after == "manifest"
            else output / f"rate-{int(crash_after[-1]):03d}" / "summary.json"
        )

        def unlink_then_crash(path: Path, *, missing_ok: bool = False) -> None:
            original_unlink(path, missing_ok=missing_ok)
            if path == target:
                raise _InjectedRetirementCrash

        monkeypatch.setattr(Path, "unlink", unlink_then_crash)

    with pytest.raises(_InjectedRetirementCrash):
        _retire_partial_finalization(
            output,
            source_sha256=source,
            rates=rates,
            rate_records=records,
        )
    monkeypatch.undo()

    runner._publish_checkpoint(stage, 1, None)
    shutil.rmtree(output)
    restored_stage = CampaignStage(
        "evaluation",
        output,
        lambda deadline: StageResult.CHECKPOINTED,
    )
    runner._restore_checkpoints(restored_stage, 2)

    progress_status = _verify_existing_progress(
        output,
        source_sha256=source,
        rates=rates,
        rate_records=records,
    )
    assert progress_status in {"partial_deadline", "in_progress"}
    _verify_orphan_rate_summaries(
        output,
        source_sha256=source,
        rates=rates,
        records=records,
        config=config,  # type: ignore[arg-type]
        progress_status=progress_status,
    )

    _retire_partial_finalization(
        output,
        source_sha256=source,
        rates=rates,
        rate_records=records,
    )
    assert not (output / "manifest.json").exists()
    assert not list(output.glob("rate-*/summary.json"))
    assert json.loads((output / "progress.json").read_text())["status"] == "in_progress"


def _write_fixed_selection(
    path: Path,
    *,
    config_path: Path,
    code_manifest_path: Path,
) -> None:
    write_canonical_json(
        path,
        {
            "evidence_role": "predeclared_selection_not_evidence",
            "pilot_rows": [],
            "selected_noise_points": [0.0375],
            "selection_mode": "fixed",
            "source_sha256": {
                "code_manifest": sha256_file(code_manifest_path),
                "config": sha256_file(config_path),
            },
        },
    )
    write_canonical_json(
        path.parent / "manifest.json",
        {
            "complete": True,
            "role": "pilot",
            "selection_sha256": sha256_file(path),
            "shards": {},
        },
    )


def test_evaluation_reuses_selection_receipt_across_bounded_batch_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = Path("configs/accuracy_disconfirm_p0375.json")
    code_manifest_path = tmp_path / "code/code.json"
    code_manifest_path.parent.mkdir()
    code_manifest_path.write_text("{}\n")
    selection_path = tmp_path / "pilot/selection.json"
    selection_path.parent.mkdir()
    _write_fixed_selection(
        selection_path,
        config_path=config_path,
        code_manifest_path=code_manifest_path,
    )
    output = tmp_path / "evaluation"
    semantic_verifications = 0
    original_verify = evaluation_module.verify_selection_publication

    def counting_verify(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal semantic_verifications
        semantic_verifications += 1
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(evaluation_module, "verify_selection_publication", counting_verify)
    first, receipt, receipt_sha256 = _selection_for_evaluation(
        selection_path=selection_path,
        config_path=config_path,
        code_manifest_path=code_manifest_path,
        output=output,
        resume=False,
    )
    assert receipt is not None
    assert receipt_sha256 is None
    output.mkdir()
    write_canonical_json(output / "selection-verification.json", receipt)
    (output / "rate-000/batch-00000").mkdir(parents=True)

    resumed, resumed_receipt, receipt_sha256 = _selection_for_evaluation(
        selection_path=selection_path,
        config_path=config_path,
        code_manifest_path=code_manifest_path,
        output=output,
        resume=True,
    )

    assert resumed == first
    assert resumed_receipt is None
    assert receipt_sha256 == sha256_file(output / "selection-verification.json")
    assert semantic_verifications == 1


def test_evaluation_resume_rejects_missing_selection_receipt(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    (output / "rate-000/batch-00000").mkdir(parents=True)
    with pytest.raises(ValueError, match="selection verification receipt"):
        _selection_for_evaluation(
            selection_path=tmp_path / "pilot/selection.json",
            config_path=Path("configs/accuracy_disconfirm_p0375.json"),
            code_manifest_path=tmp_path / "code/code.json",
            output=output,
            resume=True,
        )


def test_evaluation_recreates_missing_receipt_before_any_committed_batch(
    tmp_path: Path,
) -> None:
    config_path = Path("configs/accuracy_disconfirm_p0375.json")
    code_manifest_path = tmp_path / "code/code.json"
    code_manifest_path.parent.mkdir()
    code_manifest_path.write_text("{}\n")
    selection_path = tmp_path / "pilot/selection.json"
    selection_path.parent.mkdir()
    _write_fixed_selection(
        selection_path,
        config_path=config_path,
        code_manifest_path=code_manifest_path,
    )
    output = tmp_path / "evaluation"
    output.mkdir()

    selection, receipt, receipt_sha256 = _selection_for_evaluation(
        selection_path=selection_path,
        config_path=config_path,
        code_manifest_path=code_manifest_path,
        output=output,
        resume=True,
    )

    assert selection.selection_mode == "fixed"
    assert receipt is not None
    assert receipt_sha256 is None


def test_pilot_semantic_replay_is_bounded_across_batch_process_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
    code_manifest_path = tmp_path / "code/code.json"
    config_path.write_text("{}\n")
    code_manifest_path.parent.mkdir()
    code_manifest_path.write_text("{}\n")
    selection_path = tmp_path / "pilot/selection.json"
    output = tmp_path / "evaluation"
    verified = evaluation_module.VerifiedSelectionPublication(
        evidence_role="selection_only_not_held_out",
        manifest_sha256="a" * 64,
        rates=(0.01,),
        selection_mode="pilot",
        selection_sha256="b" * 64,
    )
    semantic_replays = 0
    receipt_checks = 0

    def replay_once(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal semantic_replays
        semantic_replays += 1
        return verified

    def check_receipt(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal receipt_checks
        receipt_checks += 1
        return verified, "c" * 64

    monkeypatch.setattr(evaluation_module, "verify_selection_publication", replay_once)
    selection, receipt, _ = _selection_for_evaluation(
        selection_path=selection_path,
        config_path=config_path,
        code_manifest_path=code_manifest_path,
        output=output,
        resume=False,
    )
    assert selection == verified
    assert receipt is not None
    output.mkdir()
    write_canonical_json(output / "selection-verification.json", receipt)
    (output / "rate-000/batch-00000").mkdir(parents=True)
    monkeypatch.setattr(
        evaluation_module,
        "verify_selection_verification_receipt",
        check_receipt,
    )

    for _ in range(8):
        resumed, _, _ = _selection_for_evaluation(
            selection_path=selection_path,
            config_path=config_path,
            code_manifest_path=code_manifest_path,
            output=output,
            resume=True,
        )
        assert resumed == verified

    assert semantic_replays == 1
    assert receipt_checks == 8
