from __future__ import annotations

import copy
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


def _write_verified_batch(
    output: Path,
    *,
    source_sha256: dict[str, object],
    rate: float = 0.01,
    shots: int = 1,
) -> _VerifiedBatch:
    arrays = _latency_batch_arrays(shots, total_latency=0.125)
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
    staging = output / "rate-000/.batch-00000.tmp"
    staging.mkdir(parents=True)
    outcomes_path = staging / "outcomes.npz"
    _write_outcomes_atomic(outcomes_path, arrays)
    write_canonical_json(
        staging / "manifest.json",
        _batch_manifest(
            rate_index=0,
            rate=rate,
            batch_index=0,
            start=0,
            arrays=arrays,
            probability_reliability=reliability,
            outcomes_path=outcomes_path,
            source_sha256=source_sha256,
        ),
    )
    committed = staging.with_name("batch-00000")
    os.replace(staging, committed)
    return _verify_batch_manifest(
        committed / "manifest.json",
        rate_index=0,
        rate=rate,
        batch_index=0,
        expected_start=0,
        expected_indices=np.arange(shots, dtype=np.int64),
        source_sha256=source_sha256,
    )


def test_atomic_json_hash_uses_the_exact_bytes_written_across_newline_translation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def translated_write_text(path: Path, data: str, *args: object, **kwargs: object) -> int:
        return path.write_bytes(data.replace("\n", "\r\n").encode())

    monkeypatch.setattr(Path, "write_text", translated_write_text)
    payload = {"selection": "receipt"}
    path = tmp_path / "selection-verification.json"

    evaluation_module._write_json_atomic(path, payload)

    assert sha256_file(path) == evaluation_module._canonical_json_sha256(payload)


def test_atomic_json_replaces_hardlinked_crash_temp_without_mutating_its_target(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("outside sentinel\n")
    output = tmp_path / "evaluation"
    output.mkdir()
    path = output / "progress.json"
    os.link(outside, output / ".progress.json.tmp")

    evaluation_module._write_json_atomic(path, {"status": "in_progress"})

    assert outside.read_text() == "outside sentinel\n"
    assert json.loads(path.read_text()) == {"status": "in_progress"}


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
    production_manifest = _batch_manifest(
        rate_index=0,
        rate=0.0375,
        batch_index=0,
        start=0,
        arrays=arrays,
        probability_reliability=reliability,
        outcomes_path=outcomes_path,
        source_sha256=source,
    )
    for method in ("soft_prior", "residual"):
        persisted_sum = float(np.sum(arrays[f"{method}_fno_latency_seconds"], dtype=np.float64))
        assert production_manifest["fno_inference_latency_seconds"][method] == (persisted_sum)
        assert persisted_sum != total_latency

    manifest = copy.deepcopy(production_manifest)
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
    first, receipt, receipt_sha256, semantics_verified = _selection_for_evaluation(
        selection_path=selection_path,
        config_path=config_path,
        code_manifest_path=code_manifest_path,
        output=output,
        resume=False,
    )
    assert receipt is not None
    assert receipt_sha256 is None
    assert semantics_verified is True
    output.mkdir()
    write_canonical_json(output / "selection-verification.json", receipt)
    receipt_sha256 = sha256_file(output / "selection-verification.json")
    source_sha256 = {"selection_verification": receipt_sha256}
    _write_verified_batch(output, source_sha256=source_sha256, rate=0.0375)

    resumed, resumed_receipt, receipt_sha256, semantics_verified = _selection_for_evaluation(
        selection_path=selection_path,
        config_path=config_path,
        code_manifest_path=code_manifest_path,
        output=output,
        resume=True,
    )
    records = evaluation_module._scan_all_batches(
        output,
        rates=(0.0375,),
        expected_indices={0: np.arange(1, dtype=np.int64)},
        source_sha256=source_sha256,
        allow_staging=True,
    )
    evaluation_module._verify_selection_semantics(
        resumed,
        records=records,
        selection_receipt_sha256=receipt_sha256,
        already_verified=semantics_verified,
        selection_path=selection_path,
        config_path=config_path,
        code_manifest_path=code_manifest_path,
    )

    assert resumed == first
    assert resumed_receipt is None
    assert receipt_sha256 == sha256_file(output / "selection-verification.json")
    assert semantics_verified is False
    assert semantic_verifications == 1


def test_evaluation_resume_rejects_missing_selection_receipt(tmp_path: Path) -> None:
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
    _, receipt, _, _ = _selection_for_evaluation(
        selection_path=selection_path,
        config_path=config_path,
        code_manifest_path=code_manifest_path,
        output=output,
        resume=False,
    )
    assert receipt is not None
    output.mkdir()
    receipt_path = output / "selection-verification.json"
    write_canonical_json(receipt_path, receipt)
    _write_verified_batch(
        output,
        source_sha256={"selection_verification": sha256_file(receipt_path)},
        rate=0.0375,
    )
    receipt_path.unlink()

    with pytest.raises(ValueError, match="selection verification receipt"):
        _selection_for_evaluation(
            selection_path=selection_path,
            config_path=config_path,
            code_manifest_path=code_manifest_path,
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

    selection, receipt, receipt_sha256, semantics_verified = _selection_for_evaluation(
        selection_path=selection_path,
        config_path=config_path,
        code_manifest_path=code_manifest_path,
        output=output,
        resume=True,
    )

    assert selection.selection_mode == "fixed"
    assert receipt is not None
    assert receipt_sha256 is None
    assert semantics_verified is True


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
        return verified, receipt_sha256

    monkeypatch.setattr(evaluation_module, "verify_selection_publication", replay_once)
    selection, receipt, _, semantics_verified = _selection_for_evaluation(
        selection_path=selection_path,
        config_path=config_path,
        code_manifest_path=code_manifest_path,
        output=output,
        resume=False,
    )
    assert selection == verified
    assert receipt is not None
    assert semantics_verified is True
    output.mkdir()
    write_canonical_json(output / "selection-verification.json", receipt)
    receipt_sha256 = sha256_file(output / "selection-verification.json")
    source_sha256 = {"selection_verification": receipt_sha256}
    _write_verified_batch(output, source_sha256=source_sha256)
    monkeypatch.setattr(
        evaluation_module,
        "verify_selection_verification_receipt",
        check_receipt,
    )

    for _ in range(8):
        resumed, _, resumed_receipt_sha256, semantics_verified = _selection_for_evaluation(
            selection_path=selection_path,
            config_path=config_path,
            code_manifest_path=code_manifest_path,
            output=output,
            resume=True,
        )
        records = evaluation_module._scan_all_batches(
            output,
            rates=(0.01,),
            expected_indices={0: np.arange(1, dtype=np.int64)},
            source_sha256=source_sha256,
            allow_staging=True,
        )
        evaluation_module._verify_selection_semantics(
            resumed,
            records=records,
            selection_receipt_sha256=resumed_receipt_sha256,
            already_verified=semantics_verified,
            selection_path=selection_path,
            config_path=config_path,
            code_manifest_path=code_manifest_path,
        )
        assert resumed == verified
        assert semantics_verified is False

    assert semantic_replays == 1
    assert receipt_checks == 8


@pytest.mark.parametrize(
    "rogue_kind",
    (
        "unexpected_rate_directory",
        "unexpected_rate_file",
        "unexpected_rate_symlink",
        "unexpected_batch_name",
        "batch_file",
        "batch_symlink",
        "empty_committed_batch",
    ),
)
def test_coordinated_rehashed_false_pilot_cannot_use_rogue_path_as_batch_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rogue_kind: str,
) -> None:
    output = tmp_path / "evaluation"
    output.mkdir()
    (output / "selection-verification.json").write_text('{"coordinated": "rehash"}\n')
    if rogue_kind.startswith("unexpected_rate"):
        rogue = output / "rate-999"
    else:
        rate_dir = output / "rate-000"
        rate_dir.mkdir()
        rogue = rate_dir / (
            "batch-marker" if rogue_kind == "unexpected_batch_name" else "batch-00000"
        )
    if rogue_kind.endswith("directory") or rogue_kind in {
        "unexpected_batch_name",
        "empty_committed_batch",
    }:
        rogue.mkdir()
    elif rogue_kind.endswith("file"):
        rogue.write_text("not a verified batch\n")
    elif rogue_kind.endswith("symlink"):
        target = tmp_path / f"{rogue_kind}-target"
        target.mkdir()
        rogue.symlink_to(target, target_is_directory=True)
    else:  # pragma: no cover - guarded by the parameter table
        raise AssertionError(f"unsupported rogue kind: {rogue_kind}")

    false_pilot = evaluation_module.VerifiedSelectionPublication(
        evidence_role="selection_only_not_held_out",
        manifest_sha256="a" * 64,
        rates=(0.01,),
        selection_mode="pilot",
        selection_sha256="b" * 64,
    )
    semantic_replays = 0

    def structurally_rehashed_receipt(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        return false_pilot, "c" * 64

    def reject_false_pilot(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal semantic_replays
        semantic_replays += 1
        raise ValueError("pilot semantic replay rejected coordinated rehash")

    monkeypatch.setattr(
        evaluation_module,
        "verify_selection_verification_receipt",
        structurally_rehashed_receipt,
    )
    monkeypatch.setattr(evaluation_module, "verify_selection_publication", reject_false_pilot)
    selection, receipt, receipt_sha256, semantics_verified = _selection_for_evaluation(
        selection_path=tmp_path / "pilot/selection.json",
        config_path=tmp_path / "config.json",
        code_manifest_path=tmp_path / "code/code.json",
        output=output,
        resume=True,
    )
    assert receipt is None
    assert semantics_verified is False

    with pytest.raises(
        ValueError,
        match="unexpected evaluation|must be a directory|incomplete evaluation batch",
    ):
        records = evaluation_module._scan_all_batches(
            output,
            rates=selection.rates,
            expected_indices={0: np.arange(1, dtype=np.int64)},
            source_sha256={"selection_verification": receipt_sha256},
            allow_staging=True,
        )
        evaluation_module._verify_selection_semantics(
            selection,
            records=records,
            selection_receipt_sha256=receipt_sha256,
            already_verified=semantics_verified,
            selection_path=tmp_path / "pilot/selection.json",
            config_path=tmp_path / "config.json",
            code_manifest_path=tmp_path / "code/code.json",
        )
    assert semantic_replays == 0


@pytest.mark.parametrize("empty_kind", ("rate_directory", "batch_staging"))
def test_empty_crash_artifact_does_not_anchor_rehashed_false_pilot_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    empty_kind: str,
) -> None:
    output = tmp_path / "evaluation"
    output.mkdir()
    (output / "selection-verification.json").write_text('{"coordinated": "rehash"}\n')
    rate_dir = output / "rate-000"
    rate_dir.mkdir()
    if empty_kind == "batch_staging":
        (rate_dir / ".batch-00000.tmp").mkdir()
    false_pilot = evaluation_module.VerifiedSelectionPublication(
        evidence_role="selection_only_not_held_out",
        manifest_sha256="a" * 64,
        rates=(0.01,),
        selection_mode="pilot",
        selection_sha256="b" * 64,
    )
    semantic_replays = 0

    monkeypatch.setattr(
        evaluation_module,
        "verify_selection_verification_receipt",
        lambda *args, **kwargs: (false_pilot, "c" * 64),
    )

    def reject_false_pilot(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal semantic_replays
        semantic_replays += 1
        raise ValueError("pilot semantic replay rejected coordinated rehash")

    monkeypatch.setattr(evaluation_module, "verify_selection_publication", reject_false_pilot)
    selection, _, receipt_sha256, semantics_verified = _selection_for_evaluation(
        selection_path=tmp_path / "pilot/selection.json",
        config_path=tmp_path / "config.json",
        code_manifest_path=tmp_path / "code/code.json",
        output=output,
        resume=True,
    )
    records = evaluation_module._scan_all_batches(
        output,
        rates=selection.rates,
        expected_indices={0: np.arange(1, dtype=np.int64)},
        source_sha256={"selection_verification": receipt_sha256},
        allow_staging=True,
    )

    with pytest.raises(ValueError, match="semantic replay rejected"):
        evaluation_module._verify_selection_semantics(
            selection,
            records=records,
            selection_receipt_sha256=receipt_sha256,
            already_verified=semantics_verified,
            selection_path=tmp_path / "pilot/selection.json",
            config_path=tmp_path / "config.json",
            code_manifest_path=tmp_path / "code/code.json",
        )
    assert records == {0: []}
    assert semantic_replays == 1


@pytest.mark.parametrize(
    ("staging_names", "derived_artifact"),
    (
        ((".batch-99999.tmp",), None),
        ((".batch-00000.tmp", ".batch-00001.tmp"), None),
        ((".batch-00000.tmp",), "rate-summary"),
        ((".batch-00000.tmp",), "final-manifest"),
    ),
)
def test_batch_scan_rejects_lifecycle_impossible_staging_artifacts(
    tmp_path: Path,
    staging_names: tuple[str, ...],
    derived_artifact: str | None,
) -> None:
    output = tmp_path / "evaluation"
    rate_dir = output / "rate-000"
    rate_dir.mkdir(parents=True)
    (output / "selection-verification.json").write_text("{}\n")
    for name in staging_names:
        (rate_dir / name).mkdir()
    if derived_artifact == "rate-summary":
        (rate_dir / "summary.json").write_text("{}\n")
    elif derived_artifact == "final-manifest":
        (output / "manifest.json").write_text("{}\n")

    with pytest.raises(ValueError, match="batch staging lifecycle"):
        evaluation_module._scan_all_batches(
            output,
            rates=(0.01,),
            expected_indices={0: np.arange(1, dtype=np.int64)},
            source_sha256={"selection_verification": "c" * 64},
            allow_staging=True,
        )


def test_batch_artifact_name_accepts_indices_beyond_five_digits() -> None:
    assert evaluation_module._indexed_artifact_name("batch-100000", prefix="batch-")


def test_batch_scan_rejects_staging_in_multiple_rates(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    (output / "selection-verification.json").parent.mkdir()
    (output / "selection-verification.json").write_text("{}\n")
    for rate_index in range(2):
        (output / f"rate-{rate_index:03d}/.batch-00000.tmp").mkdir(parents=True)

    with pytest.raises(ValueError, match="batch staging lifecycle"):
        evaluation_module._scan_all_batches(
            output,
            rates=(0.01, 0.02),
            expected_indices={
                0: np.arange(1, dtype=np.int64),
                1: np.arange(1, dtype=np.int64),
            },
            source_sha256={"selection_verification": "c" * 64},
            allow_staging=True,
        )


def test_batch_staging_must_belong_to_first_unfinished_rate(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    (output / "selection-verification.json").parent.mkdir()
    (output / "selection-verification.json").write_text("{}\n")
    (output / "rate-001/.batch-00000.tmp").mkdir(parents=True)
    records = evaluation_module._scan_all_batches(
        output,
        rates=(0.01, 0.02),
        expected_indices={
            0: np.arange(1, dtype=np.int64),
            1: np.arange(1, dtype=np.int64),
        },
        source_sha256={"selection_verification": "c" * 64},
        allow_staging=True,
    )
    config = SimpleNamespace(
        max_test_shots_per_point=1,
        target_failures=1,
        test_batch_shots=1,
        test_stopping_mode="fixed",
    )

    with pytest.raises(ValueError, match="first unfinished rate"):
        evaluation_module._verify_staging_lifecycle(
            output,
            records=records,
            config=config,
        )


def test_batch_staging_accepts_next_batch_of_first_unfinished_rate(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    output.mkdir()
    (output / "selection-verification.json").write_text("{}\n")
    source_sha256 = {"selection_verification": "c" * 64}
    _write_verified_batch(output, source_sha256=source_sha256)
    (output / "rate-001/.batch-00000.tmp").mkdir(parents=True)
    records = evaluation_module._scan_all_batches(
        output,
        rates=(0.01, 0.02),
        expected_indices={
            0: np.arange(1, dtype=np.int64),
            1: np.arange(1, dtype=np.int64),
        },
        source_sha256=source_sha256,
        allow_staging=True,
    )
    config = SimpleNamespace(
        max_test_shots_per_point=1,
        target_failures=1,
        test_batch_shots=1,
        test_stopping_mode="fixed",
    )

    evaluation_module._verify_staging_lifecycle(
        output,
        records=records,
        config=config,
    )


def test_batch_staging_rejects_next_batch_after_all_rates_stop(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    output.mkdir()
    (output / "selection-verification.json").write_text("{}\n")
    source_sha256 = {"selection_verification": "c" * 64}
    _write_verified_batch(output, source_sha256=source_sha256)
    (output / "rate-000/.batch-00001.tmp").mkdir()
    records = evaluation_module._scan_all_batches(
        output,
        rates=(0.01,),
        expected_indices={0: np.arange(1, dtype=np.int64)},
        source_sha256=source_sha256,
        allow_staging=True,
    )
    config = SimpleNamespace(
        max_test_shots_per_point=1,
        target_failures=1,
        test_batch_shots=1,
        test_stopping_mode="fixed",
    )

    with pytest.raises(ValueError, match="first unfinished rate"):
        evaluation_module._verify_staging_lifecycle(
            output,
            records=records,
            config=config,
        )


def test_evaluator_replays_unanchored_receipt_before_any_mutation_or_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SemanticReplayReached(BaseException):
        pass

    output = tmp_path / "evaluation"
    (output / "rate-000").mkdir(parents=True)
    (output / "selection-verification.json").write_text("{}\n")
    run_mode = tmp_path / "run-mode.json"
    run_mode.write_text("{}\n")
    selection = evaluation_module.VerifiedSelectionPublication(
        evidence_role="selection_only_not_held_out",
        manifest_sha256="a" * 64,
        rates=(0.0375,),
        selection_mode="pilot",
        selection_sha256="b" * 64,
    )
    shard_calls = 0

    def verified_shards(*args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal shard_calls
        shard_calls += 1
        seed = shard_calls
        return SimpleNamespace(
            indices_for_rate=lambda rate_index: np.arange(2_048, dtype=np.int64),
            manifest_sha256=f"manifest-{seed}",
            shard_manifest_sha256=[f"shard-{seed}"],
            shards=(SimpleNamespace(seed=seed),),
        )

    parameters = SimpleNamespace(alpha=1.0, beta=1.0, temperature=1.0)
    model = SimpleNamespace(checkpoint={"path": "model.pt"}, epoch=1, parameters=parameters)
    monkeypatch.setattr(
        evaluation_module,
        "verify_campaign_run_mode",
        lambda *args, **kwargs: {
            "mode": "canonical",
            "scientific_claims_permitted": True,
        },
    )
    monkeypatch.setattr(
        evaluation_module,
        "load_campaign_code",
        lambda *args, **kwargs: (
            None,
            np.zeros((1, 1), dtype=np.uint8),
            None,
            np.zeros((1, 1), dtype=np.uint8),
        ),
    )
    monkeypatch.setattr(
        evaluation_module,
        "verify_selection_verification_receipt",
        lambda *args, **kwargs: (selection, "c" * 64),
    )
    monkeypatch.setattr(evaluation_module, "load_verified_shards", verified_shards)
    monkeypatch.setattr(
        evaluation_module,
        "verify_shard_selection_provenance",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        evaluation_module,
        "_load_selected_models",
        lambda *args, **kwargs: (
            {"soft_prior": model, "residual": model},
            {"model": "verified"},
        ),
    )
    events: list[str] = []

    def semantic_replay(*args: object, **kwargs: object) -> None:
        events.append("semantic-replay")
        raise _SemanticReplayReached

    def premature_mutation_or_decode(*args: object, **kwargs: object) -> None:
        events.append("mutation-or-decode")
        raise AssertionError("evaluation mutated or decoded before semantic replay")

    monkeypatch.setattr(evaluation_module, "verify_selection_publication", semantic_replay)
    for name in (
        "_write_json_atomic",
        "_write_progress",
        "_discard_batch_staging",
        "_decode_batch",
    ):
        monkeypatch.setattr(evaluation_module, name, premature_mutation_or_decode)

    with pytest.raises(_SemanticReplayReached):
        evaluation_module.evaluate_hybrid_campaign(
            evaluation_module.EvaluationRequest(
                config=Path("configs/accuracy_disconfirm_p0375.json"),
                code=tmp_path / "code",
                run_mode=run_mode,
                selection=tmp_path / "pilot/selection.json",
                test=tmp_path / "test",
                model=tmp_path / "model",
                calibration=tmp_path / "calibration",
                out=output,
                campaign_mode="canonical",
                resume=True,
            )
        )

    assert events == ["semantic-replay"]
