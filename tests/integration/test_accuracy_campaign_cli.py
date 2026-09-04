from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from qldpc_fno.campaign.storage import LocalArtifactStore, materialize_completion

_STAGES = ("pilot", "shards", "training", "calibration", "evaluation", "summary")


def _run_campaign(
    output: Path,
    *arguments: str,
    fail_on_stage_execution: bool = False,
    stop_after_inputs: bool = False,
    environment_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CAMPAIGN_CALIBRATION_CANDIDATES": "1",
            "CAMPAIGN_CALIBRATION_SHOTS": "8",
            "CAMPAIGN_EPOCHS": "1",
            "CAMPAIGN_OUTPUT": str(output),
            "CAMPAIGN_PILOT_SHOTS": "8",
            "CAMPAIGN_REDUCED": "1",
            "CAMPAIGN_TEST_SHOTS": "8",
            "CAMPAIGN_TRAIN_SHOTS": "24",
        }
    )
    if fail_on_stage_execution:
        environment["CAMPAIGN_FAIL_ON_STAGE_EXECUTION"] = "1"
    if stop_after_inputs:
        environment["CAMPAIGN_STOP_AFTER_INPUTS"] = "1"
    if environment_overrides:
        environment.update(environment_overrides)
    return subprocess.run(
        ["bash", "scripts/run_accuracy_campaign.sh", *arguments],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=15 * 60,
    )


def _run_disconfirm_campaign(
    output: Path,
    *arguments: str,
    environment_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("CAMPAIGN_")
    }
    environment.update(
        {
            "CAMPAIGN_FAIL_ON_STAGE_EXECUTION": "1",
            "CAMPAIGN_OUTPUT": str(output),
        }
    )
    if environment_overrides:
        environment.update(environment_overrides)
    return subprocess.run(
        ["bash", "scripts/run_accuracy_campaign.sh", *arguments],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=15 * 60,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reduced_campaign_completes_refuses_overwrite_and_resumes_verified_stages(
    tmp_path: Path,
) -> None:
    output = tmp_path / "accuracy-campaign"
    canonical_config = Path("configs/accuracy_campaign.json")
    canonical_before = canonical_config.read_bytes()

    interrupted = _run_campaign(output, stop_after_inputs=True)

    assert interrupted.returncode == 0, interrupted.stderr
    assert (output / "inputs/_COMPLETE.json").is_file()
    assert not any((output / stage).exists() for stage in _STAGES)
    (output / "inputs/_COMPLETE.json").unlink()
    (output / "inputs/config.json").write_text("corrupt unpublished input")

    first = _run_campaign(output, "--resume")

    assert first.returncode == 0, first.stderr
    assert "NON-SCIENTIFIC REDUCED CAMPAIGN" in first.stderr
    assert canonical_config.read_bytes() == canonical_before
    store = LocalArtifactStore(output)
    assert store.verify_completion("inputs") is True
    recovery_completion = output / "inputs/.recovery/00000000/_COMPLETE.json"
    assert recovery_completion.is_file()
    resolved_inputs = tmp_path / "resolved-inputs"
    materialize_completion(store, "inputs", resolved_inputs)
    for stage in _STAGES:
        assert store.verify_completion(stage) is True

    mode = json.loads((resolved_inputs / "run-mode.json").read_text())
    assert mode["mode"] == "reduced_non_scientific"
    assert mode["schema_version"] == 3
    assert "bootstrap_samples" not in mode["execution_controls"]
    assert mode["scientific_claims_permitted"] is False
    assert mode["canonical_config_sha256"] == hashlib.sha256(canonical_before).hexdigest()
    assert mode["effective_config_sha256"] == _sha256(resolved_inputs / "config.json")
    assert mode["overrides"] == {
        "calibration_decode_shots_cap": 8,
        "calibration_shortlist_per_method": 1,
        "calibration_shots_cap": 8,
        "max_test_shots_per_point": 8,
        "pilot_shots_per_point": 8,
        "target_failures": 8,
        "test_batch_shots": 8,
        "training_epochs": 1,
        "train_shots_cap": 24,
    }

    results_path = output / "summary/results.json"
    markdown_path = output / "summary/results.md"
    results = json.loads(results_path.read_text())
    markdown = markdown_path.read_text()
    assert results["completion_state"] == "complete"
    assert results["evaluation_status"] == "complete"
    assert results["scientific_scope"]["code"] == "lp(3,7)_16"
    assert results["scientific_scope"]["campaign_mode"] == "reduced_non_scientific"
    assert results["scientific_scope"]["accuracy_primary"] is False
    assert results["scientific_scope"]["scientific_claims_permitted"] is False
    assert len(results["git_commit"]) == 40
    assert results["pilot"]["selection_sha256"]
    assert results["model"]["manifest_sha256"]
    assert results["calibration"]["selected_sha256"]
    assert results["held_out_test_results"]
    for row in results["held_out_test_results"]:
        assert set(row["decoders"]) == {"baseline", "soft_prior", "residual"}
        assert set(row["paired"]) == {"soft_prior", "residual"}
        for decoder in ("baseline", "soft_prior", "residual"):
            assert row["decoders"][decoder]["shots"] > 0
    assert "| Decoder | Shots | Failures |" in markdown
    assert "| Hybrid | Paired comparison status | Delta |" in markdown
    assert "Reproducibility and provenance" in markdown
    assert "NON-SCIENTIFIC REDUCED CAMPAIGN" in markdown

    completion_paths = [output / stage / "_COMPLETE.json" for stage in _STAGES]
    completion_digests = {path: _sha256(path) for path in completion_paths}
    result_digest = _sha256(results_path)

    refused = _run_campaign(output)

    assert refused.returncode == 2
    assert "refusing to overwrite existing campaign output" in refused.stderr

    resumed = _run_campaign(output, "--resume", fail_on_stage_execution=True)

    assert resumed.returncode == 0, resumed.stderr
    assert '"status": "complete"' in resumed.stdout
    assert {path: _sha256(path) for path in completion_paths} == completion_digests
    assert _sha256(results_path) == result_digest

    recovery_completion.unlink()

    rejected_mixed_campaign = _run_campaign(
        output,
        "--resume",
        environment_overrides={"CAMPAIGN_TRAIN_SHOTS": "25"},
    )

    assert rejected_mixed_campaign.returncode == 2
    assert "established campaign input publication is corrupt" in rejected_mixed_campaign.stderr
    assert not (output / "inputs/.recovery/00000001/_COMPLETE.json").exists()


def test_canonical_campaign_rejects_configuration_override(tmp_path: Path) -> None:
    output = tmp_path / "accuracy-campaign"
    environment = os.environ.copy()
    environment.update(
        {
            "CAMPAIGN_CONFIG": str(tmp_path / "alternate.json"),
            "CAMPAIGN_OUTPUT": str(output),
        }
    )

    result = subprocess.run(
        ["bash", "scripts/run_accuracy_campaign.sh"],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )

    assert result.returncode == 2
    assert "CAMPAIGN_CONFIG is not supported" in result.stderr
    assert not output.exists()


def test_disconfirm_profile_publishes_only_the_committed_fixed_config_and_resumes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "accuracy-disconfirm-p0375"

    first = _run_disconfirm_campaign(output, "--disconfirm")

    assert first.returncode == 2
    assert "completed resume unexpectedly executed stage command" in first.stderr
    store = LocalArtifactStore(output)
    assert store.verify_completion("inputs") is True
    resolved_inputs = tmp_path / "resolved-disconfirm-inputs"
    materialize_completion(store, "inputs", resolved_inputs)
    mode = json.loads((resolved_inputs / "run-mode.json").read_text())
    config = json.loads((resolved_inputs / "config.json").read_text())
    assert mode["canonical_config"] == "accuracy_disconfirm_p0375.json"
    assert mode["canonical_config_sha256"] == mode["effective_config_sha256"]
    assert mode["mode"] == "canonical"
    assert mode["overrides"] == {}
    assert config["selection_mode"] == "fixed"
    assert config["test_stopping_mode"] == "fixed"

    resume_first = _run_disconfirm_campaign(output, "--resume", "--disconfirm")
    disconfirm_first = _run_disconfirm_campaign(output, "--disconfirm", "--resume")

    for resumed in (resume_first, disconfirm_first):
        assert resumed.returncode == 2
        assert "completed resume unexpectedly executed stage command" in resumed.stderr
        assert store.verify_completion("inputs") is True


def test_disconfirm_profile_rejects_reduced_mode_arbitrary_config_and_unknown_arguments(
    tmp_path: Path,
) -> None:
    reduced_output = tmp_path / "reduced-disconfirm"
    reduced = _run_disconfirm_campaign(
        reduced_output,
        "--disconfirm",
        environment_overrides={"CAMPAIGN_REDUCED": "1"},
    )
    assert reduced.returncode == 2
    assert "--disconfirm cannot be combined with CAMPAIGN_REDUCED=1" in reduced.stderr
    assert not reduced_output.exists()

    arbitrary_output = tmp_path / "arbitrary-disconfirm"
    arbitrary = _run_disconfirm_campaign(
        arbitrary_output,
        "--disconfirm",
        environment_overrides={"CAMPAIGN_CONFIG": str(tmp_path / "alternate.json")},
    )
    assert arbitrary.returncode == 2
    assert "CAMPAIGN_CONFIG is not supported" in arbitrary.stderr
    assert not arbitrary_output.exists()

    unknown_output = tmp_path / "unknown-argument"
    unknown = _run_disconfirm_campaign(unknown_output, "--disconfirm", "--unknown")
    assert unknown.returncode == 2
    assert "usage:" in unknown.stderr
    assert not unknown_output.exists()


def test_disconfirm_profile_rejects_duplicate_arguments_and_canonical_guard(
    tmp_path: Path,
) -> None:
    for index, arguments in enumerate(
        (("--disconfirm", "--disconfirm"), ("--resume", "--resume"))
    ):
        output = tmp_path / f"duplicate-{index}"
        duplicate = _run_disconfirm_campaign(output, *arguments)
        assert duplicate.returncode == 2
        assert "usage:" in duplicate.stderr
        assert not output.exists()

    canonical_output = tmp_path / "canonical-guard"
    guarded = _run_disconfirm_campaign(canonical_output)
    assert guarded.returncode == 2
    assert "stage execution guard is available only" in guarded.stderr
    assert not canonical_output.exists()


def test_resume_rejects_established_corrupt_or_symlinked_inputs(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt"
    (corrupt / "inputs").mkdir(parents=True)
    (corrupt / "inputs/_COMPLETE.json").write_text("not json")

    rejected_corrupt = _run_campaign(corrupt, "--resume")

    assert rejected_corrupt.returncode == 2
    assert "established campaign input publication is corrupt" in rejected_corrupt.stderr

    symlinked = tmp_path / "symlinked"
    outside = tmp_path / "outside"
    symlinked.mkdir()
    outside.mkdir()
    marker = outside / "marker.txt"
    marker.write_text("unchanged")
    (symlinked / "inputs").symlink_to(outside, target_is_directory=True)

    rejected_symlink = _run_campaign(symlinked, "--resume")

    assert rejected_symlink.returncode == 2
    assert "established campaign input publication is corrupt" in rejected_symlink.stderr
    assert marker.read_text() == "unchanged"
