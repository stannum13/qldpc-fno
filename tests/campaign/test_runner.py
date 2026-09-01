from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from qldpc_fno.artifacts import sha256_file
from qldpc_fno.campaign import runner as runner_module
from qldpc_fno.campaign.runner import (
    CampaignRunner,
    CampaignStage,
    CampaignStatus,
    StageResult,
    _CampaignCommands,
    _deadline_finalization_timeout,
    _publish_partial_summary,
    _summary_markdown,
    _verified_calibration,
    _verified_model,
    _verified_pilot,
    _verify_scientific_chain,
    write_campaign_summary,
)
from qldpc_fno.campaign.storage import LocalArtifactStore, read_completion_manifest


class FakeStore:
    def __init__(self) -> None:
        self.valid: dict[str, bool] = {}
        self.manifests: dict[str, dict[str, object]] = {}
        self.publications: list[str] = []

    def exists(self, key: str) -> bool:
        return key in self.manifests

    def download(self, key: str, destination: Path) -> None:
        raise AssertionError(f"unexpected download of {key} to {destination}")

    def upload(self, source: Path, key: str) -> None:
        raise AssertionError(f"unexpected upload of {source} to {key}")

    def read_json(self, key: str) -> object:
        return self.manifests[key]

    def publish_directory(
        self,
        source: Path,
        prefix: str,
        *,
        status: str = "complete",
        deleted: tuple[str, ...] = (),
    ) -> dict[str, object]:
        self.publications.append(prefix)
        files = {
            path.relative_to(source).as_posix(): {
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
            for path in source.rglob("*")
            if path.is_file()
        }
        manifest: dict[str, object] = {
            "deleted": list(deleted),
            "files": files,
            "prefix": prefix,
            "schema_version": 1,
            "status": status,
        }
        self.manifests[f"{prefix}/_COMPLETE.json"] = manifest
        self.valid[prefix] = True
        return manifest

    def verify_completion(self, prefix: str) -> bool:
        return self.valid.get(prefix, False)


class FakeAction:
    def __init__(self, *results: StageResult) -> None:
        self.results = list(results)
        self.calls = 0

    def __call__(self, deadline_monotonic: float | None) -> StageResult:
        del deadline_monotonic
        result = self.results[self.calls]
        self.calls += 1
        return result


def _stage(
    tmp_path: Path, name: str, action: Callable[[float | None], StageResult]
) -> CampaignStage:
    directory = tmp_path / name
    directory.mkdir()
    (directory / "artifact.txt").write_text(name)
    return CampaignStage(name=name, directory=directory, run_unit=action)


def test_campaign_package_exports_runner_and_store_interfaces() -> None:
    from qldpc_fno import campaign

    assert campaign.CampaignRunner is CampaignRunner
    assert campaign.CampaignStatus is CampaignStatus
    assert campaign.ArtifactStore.__name__ == "ArtifactStore"


def test_runner_uses_the_image_bound_commit_without_git_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setenv("CAMPAIGN_GIT_COMMIT", commit)

    assert runner_module._git_commit(Path("/image-without-git")) == commit


def test_runner_rejects_an_invalid_image_bound_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAMPAIGN_GIT_COMMIT", "HEAD; touch /tmp/not-a-commit")

    with pytest.raises(ValueError, match="CAMPAIGN_GIT_COMMIT"):
        runner_module._git_commit(Path("/image-without-git"))


def test_runner_skips_only_verified_stages_and_checkpoints_bounded_units(tmp_path: Path) -> None:
    store = FakeStore()
    store.valid["pilot"] = True
    store.manifests["pilot/_COMPLETE.json"] = {
        "files": {},
        "prefix": "pilot",
        "schema_version": 1,
        "status": "complete",
    }
    pilot = FakeAction(StageResult.COMPLETE)
    training = FakeAction(StageResult.CHECKPOINTED, StageResult.COMPLETE)
    runner = CampaignRunner(
        store=store,
        stages=(
            _stage(tmp_path, "pilot", pilot),
            _stage(tmp_path, "training", training),
        ),
        checkpoint_grace_seconds=10,
        monotonic=lambda: 0.0,
    )

    status = runner.run(deadline_monotonic=100.0)

    assert status is CampaignStatus.COMPLETE
    assert pilot.calls == 0
    assert training.calls == 2
    assert store.publications == [".checkpoints/training/00000000", "training"]


def test_invalid_completion_forces_stage_rerun(tmp_path: Path) -> None:
    store = FakeStore()
    store.manifests["pilot/_COMPLETE.json"] = {"files": {"bad": {"sha256": "0"}}}
    store.valid["pilot"] = False
    pilot = FakeAction(StageResult.COMPLETE)
    runner = CampaignRunner(
        store=store,
        stages=(_stage(tmp_path, "pilot", pilot),),
        checkpoint_grace_seconds=10,
        monotonic=lambda: 0.0,
    )

    assert runner.run(deadline_monotonic=100.0) is CampaignStatus.COMPLETE
    assert pilot.calls == 1
    assert store.publications == ["pilot"]


def test_real_store_supersedes_corrupt_completion_after_rerun(tmp_path: Path) -> None:
    original = tmp_path / "original"
    original.mkdir()
    (original / "artifact.txt").write_text("verified")
    store = LocalArtifactStore(tmp_path / "store")
    store.publish_directory(original, "pilot")
    (tmp_path / "store/pilot/artifact.txt").write_text("corrupt")
    rerun = FakeAction(StageResult.COMPLETE)
    stage = _stage(tmp_path, "pilot", rerun)
    (stage.directory / "artifact.txt").write_text("verified")
    runner = CampaignRunner(
        store=store,
        stages=(stage,),
        checkpoint_grace_seconds=1,
        monotonic=lambda: 0.0,
    )

    assert runner.run(10.0) is CampaignStatus.COMPLETE
    assert rerun.calls == 1
    assert store.verify_completion("pilot") is True


def test_checkpoint_generations_upload_only_changed_files(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    directory = tmp_path / "training"
    directory.mkdir()
    (directory / "stable.bin").write_bytes(b"stable")
    calls = 0

    def unit(deadline: float | None) -> StageResult:
        nonlocal calls
        del deadline
        calls += 1
        (directory / "resume.json").write_text(json.dumps({"unit": calls}))
        return StageResult.COMPLETE if calls == 3 else StageResult.CHECKPOINTED

    runner = CampaignRunner(
        store=store,
        stages=(CampaignStage("training", directory, unit),),
        checkpoint_grace_seconds=1,
        monotonic=lambda: 0.0,
    )

    assert runner.run(100.0) is CampaignStatus.COMPLETE
    _, first = read_completion_manifest(store, ".checkpoints/training/00000000")
    _, second = read_completion_manifest(store, ".checkpoints/training/00000001")
    assert set(first["files"]) == {"resume.json", "stable.bin"}
    assert set(second["files"]) == {"resume.json"}


def test_runner_discovers_recovery_only_checkpoint_generation(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    stage = _stage(tmp_path, "training", FakeAction(StageResult.COMPLETE))
    corrupt = tmp_path / "corrupt.bin"
    corrupt.write_bytes(b"corrupt")
    store.upload(corrupt, ".checkpoints/training/00000000/artifact.txt")
    store.publish_directory(stage.directory, ".checkpoints/training/00000000")
    runner = CampaignRunner(
        store=store,
        stages=(stage,),
        checkpoint_grace_seconds=1,
        monotonic=lambda: 0.0,
    )

    assert runner._checkpoint_count(stage) == 1


def test_checkpoint_tombstone_removes_deleted_file_on_resume(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    directory = tmp_path / "training"
    directory.mkdir()
    obsolete = directory / "obsolete.bin"
    obsolete.write_bytes(b"old")
    calls = 0

    def unit(deadline: float | None) -> StageResult:
        nonlocal calls
        del deadline
        calls += 1
        if calls == 1:
            return StageResult.CHECKPOINTED
        if calls == 2:
            obsolete.unlink()
            (directory / "current.bin").write_bytes(b"new")
            return StageResult.CHECKPOINTED
        assert not obsolete.exists()
        return StageResult.COMPLETE

    runner = CampaignRunner(
        store=store,
        stages=(CampaignStage("training", directory, unit),),
        checkpoint_grace_seconds=1,
        monotonic=lambda: 0.0,
    )

    assert runner.run(100.0) is CampaignStatus.COMPLETE
    _, second = read_completion_manifest(store, ".checkpoints/training/00000001")
    assert second["deleted"] == ["obsolete.bin"]

    restored = tmp_path / "restored"
    restored.mkdir()
    materialized_stage = CampaignStage("training", restored, FakeAction(StageResult.COMPLETE))
    resumed = CampaignRunner(
        store=store,
        stages=(materialized_stage,),
        checkpoint_grace_seconds=1,
        monotonic=lambda: 0.0,
    )
    resumed._restore_checkpoints(materialized_stage, 2)
    assert not (restored / "obsolete.bin").exists()
    assert (restored / "current.bin").read_bytes() == b"new"


def test_runner_rejects_nonfinite_deadline(tmp_path: Path) -> None:
    runner = CampaignRunner(
        store=FakeStore(),
        stages=(_stage(tmp_path, "pilot", FakeAction(StageResult.COMPLETE)),),
        checkpoint_grace_seconds=10,
    )

    for deadline in (math.nan, math.inf, -math.inf):
        try:
            runner.run(deadline)
        except ValueError as error:
            assert "finite" in str(error)
        else:
            raise AssertionError("non-finite deadline was accepted")


def test_runner_stops_before_next_unit_inside_checkpoint_grace(tmp_path: Path) -> None:
    store = FakeStore()
    action = FakeAction(StageResult.CHECKPOINTED, StageResult.COMPLETE)
    times = iter((0.0, 95.0))
    stopped: list[str] = []
    runner = CampaignRunner(
        store=store,
        stages=(_stage(tmp_path, "evaluation", action),),
        checkpoint_grace_seconds=10,
        monotonic=lambda: next(times),
        on_deadline=stopped.append,
    )

    status = runner.run(deadline_monotonic=100.0)

    assert status is CampaignStatus.PARTIAL_DEADLINE
    assert action.calls == 1
    assert store.publications == [".checkpoints/evaluation/00000000"]
    assert stopped == ["deadline_before_evaluation_unit"]


def test_evaluation_finalization_reserves_grace_for_summary_and_checkpoint() -> None:
    assert _deadline_finalization_timeout(0.0) == 0.0
    assert _deadline_finalization_timeout(12.0) == 4.0


def test_subprocess_timeout_becomes_clean_partial_deadline(tmp_path: Path) -> None:
    stopped: list[str] = []

    def timeout(deadline: float | None) -> StageResult:
        del deadline
        raise subprocess.TimeoutExpired(["bounded-stage"], timeout=1.0)

    runner = CampaignRunner(
        store=FakeStore(),
        stages=(_stage(tmp_path, "calibration", timeout),),
        checkpoint_grace_seconds=10,
        monotonic=lambda: 0.0,
        on_deadline=stopped.append,
    )

    assert runner.run(100.0) is CampaignStatus.PARTIAL_DEADLINE
    assert stopped == ["deadline_during_calibration_unit"]


def test_deadline_callback_changes_are_checkpointed_for_resume(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    directory = tmp_path / "evaluation"
    directory.mkdir()
    (directory / "progress.json").write_text("{}")

    def timeout(deadline: float | None) -> StageResult:
        del deadline
        raise subprocess.TimeoutExpired(["evaluation"], timeout=1.0)

    def finalize(reason: str) -> None:
        assert reason == "deadline_during_evaluation_unit"
        (directory / "manifest.json").write_text('{"status":"partial_deadline"}')

    runner = CampaignRunner(
        store=store,
        stages=(CampaignStage("evaluation", directory, timeout),),
        checkpoint_grace_seconds=10,
        monotonic=lambda: 0.0,
        on_deadline=finalize,
    )

    assert runner.run(100.0) is CampaignStatus.PARTIAL_DEADLINE
    _, first = read_completion_manifest(store, ".checkpoints/evaluation/00000000")
    _, second = read_completion_manifest(store, ".checkpoints/evaluation/00000001")
    assert set(first["files"]) == {"progress.json"}
    assert set(second["files"]) == {"manifest.json"}


def test_training_adapter_separates_teacher_chunk_from_epoch(tmp_path: Path) -> None:
    commands_seen: list[list[str]] = []
    workspace = tmp_path / "work"

    def run(command: list[str], timeout: float | None) -> None:
        assert timeout is None
        commands_seen.append(command)
        model = workspace / "model"
        model.mkdir(parents=True, exist_ok=True)
        if len(commands_seen) == 1:
            (model / "teacher.json").write_text("{}")
        else:
            (model / "model.json").write_text("{}")

    commands = _CampaignCommands(
        config_path=tmp_path / "config.json",
        code_path=tmp_path / "code",
        workspace=workspace,
        command_runner=run,
        calibration_grid_limit=None,
        bootstrap_samples=10,
    )

    assert commands.training(None) is StageResult.CHECKPOINTED
    assert "--prepare-teacher-only" in commands_seen[0]
    assert "--max-teacher-chunks-this-run" in commands_seen[0]
    assert "--max-epochs-this-run" not in commands_seen[0]

    assert commands.training(None) is StageResult.COMPLETE
    assert "--max-epochs-this-run" in commands_seen[1]
    assert "--prepare-teacher-only" not in commands_seen[1]
    assert "--max-teacher-chunks-this-run" not in commands_seen[1]


def test_calibration_adapter_runs_one_resumable_work_unit(tmp_path: Path) -> None:
    commands_seen: list[list[str]] = []
    workspace = tmp_path / "work"
    (workspace / "shards/calibration").mkdir(parents=True)
    (workspace / "shards/calibration/manifest.json").write_text("{}")

    def run(command: list[str], timeout: float | None) -> None:
        del timeout
        commands_seen.append(command)
        output = workspace / "calibration"
        (output / "progress.json").write_text("{}")

    commands = _CampaignCommands(
        config_path=tmp_path / "config.json",
        code_path=tmp_path / "code",
        workspace=workspace,
        command_runner=run,
        calibration_grid_limit=None,
        bootstrap_samples=10,
    )

    assert commands.calibration(None) is StageResult.CHECKPOINTED
    assert "--max-work-units-this-run" in commands_seen[0]
    assert "--resume" not in commands_seen[0]
    assert commands.calibration(None) is StageResult.CHECKPOINTED
    assert "--resume" in commands_seen[1]


def test_evaluation_deadline_finalizer_uses_evaluator_deadline_path(tmp_path: Path) -> None:
    commands_seen: list[tuple[list[str], float | None]] = []
    workspace = tmp_path / "work"
    evaluation = workspace / "evaluation"
    evaluation.mkdir(parents=True)
    (evaluation / "progress.json").write_text("{}")

    def run(command: list[str], timeout: float | None) -> None:
        commands_seen.append((command, timeout))
        (evaluation / "manifest.json").write_text('{"status":"partial_deadline"}')

    commands = _CampaignCommands(
        config_path=tmp_path / "config.json",
        code_path=tmp_path / "code",
        workspace=workspace,
        command_runner=run,
        calibration_grid_limit=None,
        bootstrap_samples=10,
    )

    assert commands.finalize_evaluation(7.5) is True
    command, timeout = commands_seen[0]
    assert command[command.index("--deadline-monotonic") + 1] == "0"
    assert "--resume" in command
    assert timeout == 7.5


def test_pilot_adapter_does_not_trust_manifest_presence_alone(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    pilot = workspace / "pilot"
    pilot.mkdir(parents=True)
    (pilot / "manifest.json").write_text('{"complete": true, "role": "pilot"}')
    commands = _CampaignCommands(
        config_path=tmp_path / "config.json",
        code_path=tmp_path / "code",
        workspace=workspace,
        command_runner=lambda command, timeout: (_ for _ in ()).throw(
            AssertionError(f"unexpected command: {command}")
        ),
        calibration_grid_limit=None,
        bootstrap_samples=10,
    )

    try:
        commands.pilot(None)
    except KeyError, TypeError, ValueError:
        pass
    else:
        raise AssertionError("pilot manifest presence bypassed content verification")


def test_summary_keeps_selection_and_calibration_out_of_held_out_evidence(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    pilot = campaign / "pilot"
    calibration = campaign / "calibration"
    evaluation = campaign / "evaluation"
    pilot.mkdir(parents=True)
    calibration.mkdir()
    (evaluation / "rate-000").mkdir(parents=True)

    selection = {
        "pilot_rows": [{"block_errors": 0, "error_rate": 0.01, "shots": 8}],
        "selected_noise_points": [0.01],
        "source_sha256": {},
    }
    (pilot / "selection.json").write_text(json.dumps(selection))
    from qldpc_fno.artifacts import sha256_file, write_canonical_json

    write_canonical_json(
        pilot / "manifest.json",
        {
            "complete": True,
            "role": "pilot",
            "selection_sha256": sha256_file(pilot / "selection.json"),
        },
    )
    write_canonical_json(calibration / "grid.json", {"candidates": [{"soft_prior": {}}]})
    write_canonical_json(
        calibration / "selected.json",
        {
            "complete": True,
            "selected": {"soft_prior": {"block_errors": 0}},
            "source_role": "calibration",
            "source_sha256": {"grid": sha256_file(calibration / "grid.json")},
        },
    )
    rate_summary = {
        "accuracy_compatible": {"soft_prior": True, "residual": False},
        "decoders": {
            name: {
                "block_error_rate": 0.125,
                "block_error_rate_95ci_high": 0.47,
                "block_error_rate_95ci_low": 0.02,
                "block_errors": 1,
                "latency_seconds": {"p50": 0.01, "p95": 0.02},
                "shots": 8,
                "syndrome_valid_rate": 1.0,
            }
            for name in ("baseline", "soft_prior", "residual")
        },
        "error_rate": 0.01,
        "diagnostics": {
            "soft_prior_end_to_end_latency_seconds": {"p50": 0.02, "p95": 0.03},
            "residual_end_to_end_latency_seconds": {"p50": 0.03, "p95": 0.04},
        },
        "paired": {
            method: {
                "block_error_delta": -0.125,
                "block_error_delta_95ci_high": 0.0,
                "block_error_delta_95ci_low": -0.25,
            }
            for method in ("soft_prior", "residual")
        },
        "shots": 8,
        "source_sha256": {"test_manifest": "held-out"},
        "status": "complete",
        "stop_reason": "shot_cap",
    }
    write_canonical_json(evaluation / "rate-000/summary.json", rate_summary)
    write_canonical_json(
        evaluation / "manifest.json",
        {
            "complete": True,
            "rates": {
                "0": {
                    "error_rate": 0.01,
                    "shots": 8,
                    "status": "complete",
                    "stop_reason": "shot_cap",
                    "summary_path": "rate-000/summary.json",
                    "summary_sha256": sha256_file(evaluation / "rate-000/summary.json"),
                }
            },
            "selected_calibration": {},
            "source_sha256": {"test_manifest": "held-out"},
            "status": "complete",
        },
    )

    output = campaign / "summary"
    with pytest.raises(ValueError, match="requires config_path and code_path"):
        write_campaign_summary(
            campaign,
            output,
            completion_state=CampaignStatus.PARTIAL_DEADLINE,
            git_commit="abc123",
            early_stop_reasons=(),
        )

    assert _verified_pilot(campaign)["evidence_role"] == "selection_only_not_held_out"
    assert _verified_calibration(campaign)["evidence_role"] == "tuning_only_not_held_out"
    markdown = _summary_markdown(
        {
            "calibration": {
                "grid_sha256": "grid-hash",
                "selected_sha256": "selection-hash",
            },
            "completion_state": "partial_deadline",
            "early_stop_reasons": ["campaign_deadline"],
            "git_commit": "abc123",
            "held_out_test_results": [rate_summary],
            "model": {
                "manifest_sha256": "model-manifest-hash",
                "model_sha256": "model-hash",
            },
            "pilot": {"selected_noise_points": [0.01]},
        }
    )
    assert "Held-out test results" in markdown
    assert "95% Wilson interval" in markdown
    assert "Paired block-error delta" in markdown
    assert "Syndrome-valid rate" in markdown
    assert "Timing diagnostics" in markdown
    assert "Calibration is not held-out evidence" in markdown
    assert "abc123" in markdown
    assert "grid-hash" in markdown
    assert "model-hash" in markdown
    assert "campaign_deadline" in markdown


def test_complete_summary_rejects_missing_scientific_stages(tmp_path: Path) -> None:
    try:
        write_campaign_summary(
            tmp_path / "campaign",
            tmp_path / "summary",
            completion_state=CampaignStatus.COMPLETE,
            git_commit="abc123",
            early_stop_reasons=(),
        )
    except ValueError as error:
        assert "complete summary requires" in str(error)
    else:
        raise AssertionError("missing campaign stages were labeled complete")


def test_summary_rejects_model_checkpoint_path_escape(tmp_path: Path) -> None:
    from qldpc_fno.artifacts import sha256_file, write_canonical_json

    campaign = tmp_path / "campaign"
    model = campaign / "model"
    model.mkdir(parents=True)
    (model / "model.pt").write_bytes(b"model")
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"checkpoint")
    write_canonical_json(
        model / "model.json",
        {
            "checkpoints": [{"path": str(outside), "sha256": sha256_file(outside)}],
            "complete": True,
            "epoch": 1,
            "git_commit": "abc123",
            "sha256": sha256_file(model / "model.pt"),
            "source_role": "train",
        },
    )

    with pytest.raises(ValueError, match="safe relative"):
        _verified_model(campaign)


def test_summary_rejects_fixed_model_symlink_escape(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    model = campaign / "model"
    model.mkdir(parents=True)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"model")
    (model / "model.pt").symlink_to(outside)
    (model / "model.json").write_text(
        json.dumps(
            {
                "checkpoints": [],
                "complete": True,
                "epoch": 1,
                "git_commit": "abc123",
                "sha256": "unused",
                "source_role": "train",
            }
        )
    )

    with pytest.raises(ValueError, match="symlink"):
        _verified_model(campaign)


@pytest.mark.parametrize("extra_rate_role", ["train", "calibration", "test", "evaluation"])
def test_scientific_chain_requires_exact_selected_rate_coordinates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_rate_role: str,
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}")
    code = tmp_path / "code"
    code.mkdir()
    (code / "code.json").write_text("{}")
    campaign = tmp_path / "campaign"
    for directory in ("pilot", "model", "calibration", "evaluation"):
        (campaign / directory).mkdir(parents=True)
    selection = {
        "selected_noise_points": [0.01],
        "source_sha256": {
            "code_manifest": sha256_file(code / "code.json"),
            "config": sha256_file(config),
        },
    }
    (campaign / "pilot/selection.json").write_text(json.dumps(selection))

    shard_sets: dict[str, SimpleNamespace] = {}
    for role, seed in (("train", 1), ("calibration", 2), ("test", 3)):
        shards = [SimpleNamespace(rate_index=0, error_rate=0.01, seed=seed)]
        if extra_rate_role == role:
            shards.append(SimpleNamespace(rate_index=1, error_rate=0.02, seed=seed + 10))
        shard_sets[role] = SimpleNamespace(
            manifest_sha256=f"{role}-manifest",
            shard_manifest_sha256=[f"{role}-shard"],
            shards=shards,
        )

    monkeypatch.setattr(
        runner_module.CampaignConfig,
        "from_json",
        classmethod(lambda cls, path: object()),
    )
    monkeypatch.setattr(runner_module, "load_campaign_code", lambda path: None)
    monkeypatch.setattr(
        runner_module,
        "load_verified_shards",
        lambda path, **kwargs: shard_sets[path.name],
    )

    model_sources = {
        "code_manifest": sha256_file(code / "code.json"),
        "config": sha256_file(config),
        "train_manifest": "train-manifest",
        "train_shard_manifests": ["train-shard"],
    }
    (campaign / "model/model.json").write_text(
        json.dumps({"git_commit": "abc123", "source_sha256": model_sources})
    )
    (campaign / "calibration/grid.json").write_text("{}")
    calibration_sources = {
        "calibration_manifest": "calibration-manifest",
        "calibration_shard_manifests": ["calibration-shard"],
        "code_manifest": sha256_file(code / "code.json"),
        "config": sha256_file(config),
        "grid": sha256_file(campaign / "calibration/grid.json"),
        "model_manifest": sha256_file(campaign / "model/model.json"),
    }
    (campaign / "calibration/selected.json").write_text(
        json.dumps({"source_role": "calibration", "source_sha256": calibration_sources})
    )
    evaluation_sources = {
        "calibration_grid": sha256_file(campaign / "calibration/grid.json"),
        "calibration_manifest": "calibration-manifest",
        "calibration_selected": sha256_file(campaign / "calibration/selected.json"),
        "calibration_shard_manifests": ["calibration-shard"],
        "code_manifest": sha256_file(code / "code.json"),
        "config": sha256_file(config),
        "model_manifest": sha256_file(campaign / "model/model.json"),
        "selection": sha256_file(campaign / "pilot/selection.json"),
        "test_manifest": "test-manifest",
        "test_shard_manifests": ["test-shard"],
    }
    rates = {"0": {"error_rate": 0.01}}
    if extra_rate_role == "evaluation":
        rates["1"] = {"error_rate": 0.02}
    (campaign / "evaluation/manifest.json").write_text(
        json.dumps({"rates": rates, "source_sha256": evaluation_sources})
    )

    with pytest.raises(ValueError, match="rates do not match"):
        _verify_scientific_chain(
            campaign,
            config_path=config,
            code_path=code,
            git_commit="abc123",
        )


def test_deadline_summaries_are_versioned_and_do_not_complete_summary_stage(
    tmp_path: Path,
) -> None:
    store = FakeStore()

    first = _publish_partial_summary(store, tmp_path, "abc123", "deadline_before_training_unit")
    second = _publish_partial_summary(store, tmp_path, "abc123", "deadline_before_training_unit")

    assert first == ".partial-summaries/00000000"
    assert second == ".partial-summaries/00000001"
    assert store.publications == [first, second]
    assert store.verify_completion("summary") is False


def test_deadline_summary_versioning_resolves_recovery_only_completion(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    conflict = tmp_path / "conflict.json"
    conflict.write_text('{"corrupt":true}')
    store.upload(conflict, ".partial-summaries/00000000/results.json")

    first = _publish_partial_summary(
        store,
        tmp_path / "first-workspace",
        "abc123",
        "deadline_before_training_unit",
    )
    second = _publish_partial_summary(
        store,
        tmp_path / "second-workspace",
        "abc123",
        "deadline_before_training_unit",
    )

    assert first == ".partial-summaries/00000000"
    assert second == ".partial-summaries/00000001"
