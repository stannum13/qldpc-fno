from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

COMMIT = "0123456789abcdef0123456789abcdef01234567"
CAMPAIGN_ID = "accuracy-20260901-010203-a1b2c3"


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(0o755)


def _fake_tools(tmp_path: Path) -> tuple[Path, Path]:
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir(parents=True)
    log = tmp_path / "gcloud.jsonl"
    _write_executable(
        binary_dir / "git",
        """#!/usr/bin/env python3
import os
import sys

arguments = sys.argv[1:]
if arguments == ["rev-parse", "--show-toplevel"]:
    print(os.environ["FAKE_GIT_ROOT"])
elif arguments == ["rev-parse", "HEAD"]:
    print(os.environ["FAKE_GIT_COMMIT"])
elif arguments == ["hash-object", "--stdin"]:
    print("abcdef0123456789abcdef0123456789abcdef01")
elif arguments == ["status", "--porcelain=v1", "--untracked-files=all"]:
    print(os.environ.get("FAKE_GIT_STATUS", ""), end="")
else:
    raise SystemExit(f"unexpected git command: {arguments}")
""",
    )
    _write_executable(
        binary_dir / "gcloud",
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

arguments = sys.argv[1:]
with Path(os.environ["FAKE_GCLOUD_LOG"]).open("a") as handle:
    handle.write(json.dumps(arguments) + "\\n")
if arguments == ["config", "get-value", "project"]:
    print(os.environ.get("FAKE_GCLOUD_PROJECT", "science-project"))
    raise SystemExit(0)

existing = os.environ.get("FAKE_GCLOUD_EXISTING", "")
describe_error = os.environ.get("FAKE_GCLOUD_DESCRIBE_ERROR") == "1"
if arguments[:3] == ["artifacts", "repositories", "describe"]:
    if describe_error:
        print("PERMISSION_DENIED", file=sys.stderr)
    elif existing != "repository":
        print("NOT_FOUND", file=sys.stderr)
    raise SystemExit(0 if existing == "repository" else 1)
if arguments[:3] == ["storage", "buckets", "describe"]:
    if describe_error:
        print("PERMISSION_DENIED", file=sys.stderr)
    elif existing != "bucket":
        print("NOT_FOUND", file=sys.stderr)
    raise SystemExit(0 if existing == "bucket" else 1)
if arguments[:3] == ["run", "jobs", "describe"]:
    if describe_error:
        print("PERMISSION_DENIED", file=sys.stderr)
    elif existing != "job":
        print("NOT_FOUND", file=sys.stderr)
    raise SystemExit(0 if existing == "job" else 1)
if arguments[:3] == ["iam", "service-accounts", "describe"]:
    if describe_error:
        print("PERMISSION_DENIED", file=sys.stderr)
    elif existing != "service-account":
        print("NOT_FOUND", file=sys.stderr)
    raise SystemExit(0 if existing == "service-account" else 1)
raise SystemExit(0)
""",
    )
    return binary_dir, log


def _launch(
    tmp_path: Path,
    *arguments: str,
    environment_overrides: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    binary_dir, log = _fake_tools(tmp_path)
    environment = os.environ.copy()
    environment.update(
        {
            "CAMPAIGN_ID": CAMPAIGN_ID,
            "CLOUD_REGION": "us-central1",
            "FAKE_GCLOUD_LOG": str(log),
            "FAKE_GIT_COMMIT": COMMIT,
            "FAKE_GIT_ROOT": str(Path.cwd()),
            "PATH": f"{binary_dir}:{environment['PATH']}",
        }
    )
    if environment_overrides:
        environment.update(environment_overrides)
    result = subprocess.run(
        ["bash", "scripts/launch_cloud_campaign.sh", *arguments],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, calls


def test_dry_run_resolves_every_resource_and_performs_no_mutation(tmp_path: Path) -> None:
    result, calls = _launch(tmp_path)

    assert result.returncode == 0, result.stderr
    assert calls == [["config", "get-value", "project"]]
    expected = {
        "mode": "dry-run",
        "project": "science-project",
        "region": "us-central1",
        "repository": f"qldpc-fno-{CAMPAIGN_ID}",
        "image": (
            f"us-central1-docker.pkg.dev/science-project/qldpc-fno-{CAMPAIGN_ID}"
            f"/accuracy-campaign:{COMMIT}"
        ),
        "bucket": f"science-project-{CAMPAIGN_ID}",
        "prefix": f"campaigns/{CAMPAIGN_ID}/{COMMIT}",
        "store": (
            f"gs://science-project-{CAMPAIGN_ID}/campaigns/{CAMPAIGN_ID}/{COMMIT}"
        ),
        "cpu": "8",
        "memory": "32Gi",
        "timeout": "8h",
        "retries": "0",
        "tasks": "1",
        "git_commit": COMMIT,
        "service_account": (
            "qfno-abcdef0123456789abcdef01@science-project.iam.gserviceaccount.com"
        ),
    }
    for key, value in expected.items():
        assert f"{key}={value}" in result.stdout
    assert "gcloud artifacts repositories create" in result.stdout
    assert "gcloud storage buckets create" in result.stdout
    assert "gcloud builds submit" in result.stdout
    assert "gcloud run jobs create" in result.stdout
    assert "gcloud run jobs execute" in result.stdout
    assert "cleanup commands (not executed):" in result.stdout


def test_execute_creates_one_bounded_cpu_job_and_returns_after_submission(tmp_path: Path) -> None:
    result, calls = _launch(tmp_path, "--execute")

    assert result.returncode == 0, result.stderr
    assert calls[0] == ["config", "get-value", "project"]
    mutation_calls = [
        call
        for call in calls
        if "describe" not in call and call != ["config", "get-value", "project"]
    ]
    assert [call[:3] for call in mutation_calls] == [
        ["artifacts", "repositories", "create"],
        ["storage", "buckets", "create"],
        ["iam", "service-accounts", "create"],
        ["storage", "buckets", "add-iam-policy-binding"],
        ["storage", "buckets", "add-iam-policy-binding"],
        ["builds", "submit", "--tag"],
        ["run", "jobs", "create"],
        ["run", "jobs", "execute"],
    ]
    create_job = next(call for call in calls if call[:3] == ["run", "jobs", "create"])
    assert "--cpu=8" in create_job
    assert "--memory=32Gi" in create_job
    assert "--task-timeout=8h" in create_job
    assert "--max-retries=0" in create_job
    assert "--tasks=1" in create_job
    assert not any(argument.startswith("--gpu") for argument in create_job)
    env_argument = next(argument for argument in create_job if argument.startswith("--set-env-vars="))
    assert f"CAMPAIGN_BUCKET=science-project-{CAMPAIGN_ID}" in env_argument
    assert f"CAMPAIGN_PREFIX=campaigns/{CAMPAIGN_ID}/{COMMIT}" in env_argument
    assert f"CAMPAIGN_GIT_COMMIT={COMMIT}" in env_argument
    assert "CAMPAIGN_CONFIG=/app/configs/accuracy_campaign.json" in env_argument
    assert (
        "--service-account=qfno-abcdef0123456789abcdef01@science-project.iam.gserviceaccount.com"
        in create_job
    )
    bucket_bindings = [
        call
        for call in calls
        if call[:3] == ["storage", "buckets", "add-iam-policy-binding"]
    ]
    assert {argument for call in bucket_bindings for argument in call if argument.startswith("--role=")} == {
        "--role=roles/storage.objectCreator",
        "--role=roles/storage.objectViewer",
    }
    assert not any("roles/storage.objectAdmin" in argument for call in calls for argument in call)
    execute_job = next(call for call in calls if call[:3] == ["run", "jobs", "execute"])
    assert "--async" in execute_job
    assert "--wait" not in execute_job
    assert not any("delete" in call for call in calls)


def test_reduced_execute_uses_non_scientific_config_and_waits(tmp_path: Path) -> None:
    result, calls = _launch(tmp_path, "--execute", "--reduced")

    assert result.returncode == 0, result.stderr
    assert "campaign_mode=reduced_non_scientific" in result.stdout
    create_job = next(call for call in calls if call[:3] == ["run", "jobs", "create"])
    env_argument = next(argument for argument in create_job if argument.startswith("--set-env-vars="))
    assert "CAMPAIGN_CONFIG=/app/configs/accuracy_campaign_cloud_reduced.json" in env_argument
    assert (
        "--args=--campaign-mode=reduced_non_scientific,--calibration-grid-limit=1,"
        "--bootstrap-samples=100" in create_job
    )
    execute_job = next(call for call in calls if call[:3] == ["run", "jobs", "execute"])
    assert "--wait" in execute_job
    assert "--async" not in execute_job


@pytest.mark.parametrize("existing", ["repository", "bucket", "job", "service-account"])
def test_execute_refuses_existing_campaign_resources_before_mutation(
    tmp_path: Path, existing: str
) -> None:
    result, calls = _launch(
        tmp_path,
        "--execute",
        environment_overrides={"FAKE_GCLOUD_EXISTING": existing},
    )

    assert result.returncode == 2
    assert f"campaign {existing} already exists" in result.stderr
    assert not any(
        call[:3]
        in (
            ["artifacts", "repositories", "create"],
            ["storage", "buckets", "create"],
            ["run", "jobs", "create"],
        )
        or call[:2] == ["builds", "submit"]
        for call in calls
    )


def test_execute_fails_closed_when_resource_absence_cannot_be_verified(tmp_path: Path) -> None:
    result, calls = _launch(
        tmp_path,
        "--execute",
        environment_overrides={"FAKE_GCLOUD_DESCRIBE_ERROR": "1"},
    )

    assert result.returncode == 2
    assert "cannot verify campaign repository absence" in result.stderr
    assert not any("create" in call or call[:2] == ["builds", "submit"] for call in calls)


def test_launcher_rejects_dirty_source_and_shell_injection(tmp_path: Path) -> None:
    dirty, dirty_calls = _launch(
        tmp_path / "dirty",
        environment_overrides={"FAKE_GIT_STATUS": " M Dockerfile\n"},
    )

    assert dirty.returncode == 2
    assert "clean Git checkout" in dirty.stderr
    assert dirty_calls == []

    marker = tmp_path / "injected"
    injected, injected_calls = _launch(
        tmp_path / "injection",
        environment_overrides={"CLOUD_REGION": f"us-central1;touch {marker}"},
    )

    assert injected.returncode == 2
    assert "CLOUD_REGION" in injected.stderr
    assert not marker.exists()
    assert injected_calls == []


def test_container_is_pinned_bounded_and_starts_the_campaign_runner() -> None:
    dockerfile = Path("Dockerfile").read_text()
    ignored = Path(".dockerignore").read_text().splitlines()

    assert "ghcr.io/astral-sh/uv:0.9.17" in dockerfile
    assert "uv sync --frozen --no-dev" in dockerfile
    assert "experiments/01_build_lp_codes.py" in dockerfile
    assert "experiments/02_validate_lp_codes.py" in dockerfile
    assert 'CAMPAIGN_CONFIG="/app/configs/accuracy_campaign.json"' in dockerfile
    assert 'CAMPAIGN_CODE="/app/campaign-code"' in dockerfile
    assert 'ENTRYPOINT ["/app/.venv/bin/python", "-m", "qldpc_fno.campaign.runner"]' in dockerfile
    assert ".git" in ignored
    assert ".venv" in ignored
    assert "artifacts" in ignored
    assert "tests" in ignored
    reduced_config = json.loads(Path("configs/accuracy_campaign_cloud_reduced.json").read_text())
    assert reduced_config["pilot_shots_per_point"] == 8
    assert reduced_config["train_shots_cap"] == 24
    assert reduced_config["calibration_shots_cap"] == 8
    assert reduced_config["max_test_shots_per_point"] == 8
    assert reduced_config["training_epochs"] == 1
    for consumer in (
        Path("experiments/15_train_conditional_fno.py"),
        Path("experiments/16_calibrate_hybrid_priors.py"),
        Path("src/qldpc_fno/campaign/evaluation.py"),
        Path("src/qldpc_fno/campaign/runner.py"),
    ):
        assert "resolve_git_commit" in consumer.read_text()
