from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

COMMIT = "0123456789abcdef0123456789abcdef01234567"
CAMPAIGN_ID = "accuracy-20260901-010203-a1b2c3"
DIGEST = f"sha256:{'1' * 64}"
PROJECT_NUMBER = "888484963419"
BUILD_SERVICE_ACCOUNT = f"{PROJECT_NUMBER}-compute@developer.gserviceaccount.com"


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
import subprocess
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
elif arguments and arguments[0] == "archive":
    archive_arguments = [
        "HEAD" if argument == os.environ["FAKE_GIT_COMMIT"] else argument
        for argument in arguments
    ]
    subprocess.run(
        ["/usr/bin/git", "-C", os.environ["FAKE_GIT_ROOT"], *archive_arguments],
        check=True,
    )
else:
    raise SystemExit(f"unexpected git command: {arguments}")
""",
    )
    _write_executable(
        binary_dir / "gcloud",
        """#!/usr/bin/env python3
import json
import hashlib
import os
import sys
import tarfile
from pathlib import Path

DIGEST = "sha256:" + "1" * 64
PROJECT_NUMBER = "888484963419"


def canonical_bytes(payload):
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\\n").encode()


def record(path):
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}

arguments = sys.argv[1:]
with Path(os.environ["FAKE_GCLOUD_LOG"]).open("a") as handle:
    handle.write(json.dumps(arguments) + "\\n")
if arguments[:2] == ["builds", "submit"]:
    with tarfile.open(arguments[-1]) as archive:
        Path(os.environ["FAKE_ARCHIVE_MEMBERS"]).write_text(
            "\\n".join(sorted(archive.getnames()))
        )
if arguments == ["config", "get-value", "project"]:
    print(os.environ.get("FAKE_GCLOUD_PROJECT", "science-project"))
    raise SystemExit(0)
if arguments[:2] == ["builds", "get-default-service-account"]:
    if os.environ.get("FAKE_DEFAULT_BUILD_SA_UNSUPPORTED") == "1":
        print("Invalid choice: get-default-service-account", file=sys.stderr)
        raise SystemExit(1)
    if os.environ.get("FAKE_DEFAULT_BUILD_SA_FAILURE") == "1":
        print("PERMISSION_DENIED", file=sys.stderr)
        raise SystemExit(1)
    if "--format=value(serviceAccountEmail)" not in arguments:
        print("missing structured service-account format", file=sys.stderr)
        raise SystemExit(1)
    account = os.environ.get(
        "FAKE_DEFAULT_BUILD_SA",
        f"projects/science-project/serviceAccounts/{PROJECT_NUMBER}-compute@developer.gserviceaccount.com",
    )
    print(account)
    raise SystemExit(0)
if arguments[:2] == ["projects", "describe"]:
    if os.environ.get("FAKE_PROJECT_NUMBER_FAILURE") == "1":
        print("PERMISSION_DENIED", file=sys.stderr)
        raise SystemExit(1)
    print(os.environ.get("FAKE_PROJECT_NUMBER", PROJECT_NUMBER))
    raise SystemExit(0)
if arguments[:4] == ["artifacts", "docker", "images", "describe"]:
    print(DIGEST)
    raise SystemExit(0)
if arguments[:3] == ["artifacts", "repositories", "add-iam-policy-binding"]:
    if os.environ.get("FAKE_BUILD_WRITER_GRANT_FAILURE") == "1":
        print("PERMISSION_DENIED", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(0)

if arguments[:3] == ["storage", "cp", "--recursive"]:
    destination = Path(arguments[-1]) / "inputs"
    (destination / "code").mkdir(parents=True)
    canonical_config = Path(os.environ["FAKE_GIT_ROOT"]) / "configs/accuracy_campaign.json"
    (destination / "config.json").write_bytes(canonical_config.read_bytes())
    (destination / "code/code.json").write_bytes(canonical_bytes({"name": "lp_3_7_16"}))
    image = (
        "us-central1-docker.pkg.dev/science-project/"
        f"qldpc-fno-{os.environ['CAMPAIGN_ID']}/accuracy-campaign@{DIGEST}"
    )
    identity = {
        "bucket": f"science-project-{os.environ['CAMPAIGN_ID']}",
        "finalization_reserve_seconds": 2700,
        "image": image,
        "image_digest": DIGEST,
        "job": f"qldpc-fno-{os.environ['CAMPAIGN_ID']}",
        "kind": "cloud",
        "outer_timeout_seconds": 28800,
        "prefix": f"campaigns/{os.environ['CAMPAIGN_ID']}/{os.environ['FAKE_GIT_COMMIT']}",
        "project": "science-project",
        "region": "us-central1",
        "service_account": (
            "qfno-abcdef0123456789abcdef01@science-project.iam.gserviceaccount.com"
        ),
        "store": (
            f"gs://science-project-{os.environ['CAMPAIGN_ID']}/campaigns/"
            f"{os.environ['CAMPAIGN_ID']}/{os.environ['FAKE_GIT_COMMIT']}"
        ),
        "work_cutoff_seconds": 26100,
    }
    mode = {
        "canonical_config": "accuracy_campaign.json",
        "canonical_config_sha256": hashlib.sha256(canonical_config.read_bytes()).hexdigest(),
        "code_manifest_sha256": record(destination / "code/code.json")["sha256"],
        "effective_config_sha256": record(destination / "config.json")["sha256"],
        "execution_controls": {"calibration_grid_limit": None},
        "execution_identity": identity,
        "git_commit": os.environ["FAKE_GIT_COMMIT"],
        "mode": "canonical",
        "overrides": {},
        "schema_version": 3,
        "scientific_claims_permitted": True,
    }
    (destination / "run-mode.json").write_bytes(canonical_bytes(mode))
    files = {
        relative: record(destination / relative)
        for relative in ("code/code.json", "config.json", "run-mode.json")
    }
    completion = {
        "deleted": [],
        "files": files,
        "prefix": "inputs",
        "publication_id": "fake-input-publication",
        "schema_version": 1,
        "status": "complete",
    }
    (destination / "_COMPLETE.json").write_bytes(canonical_bytes(completion))
    raise SystemExit(0)

existing = os.environ.get("FAKE_GCLOUD_EXISTING", "")
resume = os.environ.get("FAKE_GCLOUD_RESUME") == "1"
describe_error = os.environ.get("FAKE_GCLOUD_DESCRIBE_ERROR") == "1"
if arguments[:3] == ["artifacts", "repositories", "describe"]:
    if describe_error:
        print("PERMISSION_DENIED", file=sys.stderr)
    elif existing != "repository":
        print("NOT_FOUND", file=sys.stderr)
    raise SystemExit(0 if resume or existing == "repository" else 1)
if arguments[:3] == ["storage", "buckets", "describe"]:
    if describe_error:
        print("PERMISSION_DENIED", file=sys.stderr)
    elif existing != "bucket":
        print("NOT_FOUND", file=sys.stderr)
    raise SystemExit(0 if resume or existing == "bucket" else 1)
if arguments[:3] == ["run", "jobs", "describe"]:
    if "--format=json" in arguments:
        campaign_id = os.environ["CAMPAIGN_ID"]
        commit = os.environ["FAKE_GIT_COMMIT"]
        bucket = f"science-project-{campaign_id}"
        prefix = f"campaigns/{campaign_id}/{commit}"
        service_account = os.environ.get(
            "FAKE_JOB_SERVICE_ACCOUNT",
            "qfno-abcdef0123456789abcdef01@science-project.iam.gserviceaccount.com",
        )
        store = os.environ.get("FAKE_JOB_STORE", f"gs://{bucket}/{prefix}")
        environment = {
            "CAMPAIGN_BUCKET": bucket,
            "CAMPAIGN_CALIBRATION_GRID_LIMIT": "",
            "CAMPAIGN_CANONICAL_CONFIG": "/app/configs/accuracy_campaign.json",
            "CAMPAIGN_CLOUD_JOB": f"qldpc-fno-{campaign_id}",
            "CAMPAIGN_CLOUD_PROJECT": "science-project",
            "CAMPAIGN_CLOUD_REGION": "us-central1",
            "CAMPAIGN_CODE": "/app/campaign-code",
            "CAMPAIGN_CONFIG": "/app/configs/accuracy_campaign.json",
            "CAMPAIGN_FINALIZATION_RESERVE_SECONDS": "2700",
            "CAMPAIGN_GIT_COMMIT": commit,
            "CAMPAIGN_IMAGE": os.environ["FAKE_JOB_IMAGE"],
            "CAMPAIGN_IMAGE_DIGEST": DIGEST,
            "CAMPAIGN_MODE": "canonical",
            "CAMPAIGN_OUTER_TIMEOUT_SECONDS": "28800",
            "CAMPAIGN_PREFIX": prefix,
            "CAMPAIGN_SERVICE_ACCOUNT": (
                "qfno-abcdef0123456789abcdef01@science-project.iam.gserviceaccount.com"
            ),
            "CAMPAIGN_STORE": store,
            "CAMPAIGN_WORKDIR": "/tmp/qldpc-fno-work",
            "CAMPAIGN_WORK_CUTOFF_SECONDS": "26100",
        }
        payload = {
            "metadata": {
                "labels": {
                    "qldpc-fno-identity": os.environ["FAKE_JOB_IDENTITY"],
                    "qldpc-fno-mode": "canonical",
                }
            },
            "spec": {
                "template": {
                    "spec": {
                        "parallelism": 1,
                        "taskCount": 1,
                        "template": {
                            "spec": {
                                "containers": [
                                    {
                                        "env": [
                                            {"name": name, "value": value}
                                            for name, value in sorted(environment.items())
                                        ],
                                        "image": os.environ["FAKE_JOB_IMAGE"],
                                        "resources": {
                                            "limits": {"cpu": "8", "memory": "32Gi"}
                                        },
                                    }
                                ],
                                "maxRetries": 0,
                                "serviceAccountName": service_account,
                                "timeoutSeconds": os.environ.get(
                                    "FAKE_JOB_TIMEOUT_SECONDS", "28800"
                                ),
                            }
                        },
                    }
                }
            },
        }
        print(json.dumps(payload))
        raise SystemExit(0)
    if describe_error:
        print("PERMISSION_DENIED", file=sys.stderr)
    elif existing != "job":
        print("NOT_FOUND", file=sys.stderr)
    raise SystemExit(0 if resume or existing == "job" else 1)
if arguments[:3] == ["iam", "service-accounts", "describe"]:
    if arguments[3].endswith(("@cloudbuild.gserviceaccount.com", "@developer.gserviceaccount.com")):
        if os.environ.get("FAKE_BUILD_SA_DESCRIBE_FAILURE") == "1":
            print("NOT_FOUND", file=sys.stderr)
            raise SystemExit(1)
        print(os.environ.get("FAKE_DESCRIBED_BUILD_SA", arguments[3]))
        raise SystemExit(0)
    if describe_error:
        print("PERMISSION_DENIED", file=sys.stderr)
    elif existing != "service-account":
        print("NOT_FOUND", file=sys.stderr)
    raise SystemExit(0 if resume or existing == "service-account" else 1)
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
            "FAKE_ARCHIVE_MEMBERS": str(tmp_path / "archive-members.txt"),
            "FAKE_GIT_COMMIT": COMMIT,
            "FAKE_GIT_ROOT": str(Path.cwd()),
            "FAKE_JOB_IDENTITY": "abcdef0123456789abcdef0123456789abcdef01",
            "FAKE_JOB_IMAGE": (
                f"us-central1-docker.pkg.dev/science-project/qldpc-fno-{CAMPAIGN_ID}"
                f"/accuracy-campaign@{DIGEST}"
            ),
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
    assert calls == [
        ["config", "get-value", "project"],
        [
            "builds",
            "get-default-service-account",
            "--project=science-project",
            "--format=value(serviceAccountEmail)",
        ],
        ["projects", "describe", "science-project", "--format=value(projectNumber)"],
        [
            "iam",
            "service-accounts",
            "describe",
            BUILD_SERVICE_ACCOUNT,
            "--project=science-project",
            "--format=value(email)",
        ],
    ]
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
        "canonical_execution_gate": "blocked_representative_decoder_benchmark",
        "git_commit": COMMIT,
        "service_account": (
            "qfno-abcdef0123456789abcdef01@science-project.iam.gserviceaccount.com"
        ),
        "build_service_account": BUILD_SERVICE_ACCOUNT,
    }
    for key, value in expected.items():
        assert f"{key}={value}" in result.stdout
    assert "gcloud artifacts repositories create" in result.stdout
    assert "gcloud artifacts repositories add-iam-policy-binding" in result.stdout
    assert "gcloud storage buckets create" in result.stdout
    assert "gcloud builds submit" in result.stdout
    assert "gcloud run jobs create" in result.stdout
    assert "gcloud run jobs execute" in result.stdout
    assert "CLOUD_PROJECT=science-project" in result.stdout
    assert f"CAMPAIGN_ID={CAMPAIGN_ID}" in result.stdout
    assert "launch_cloud_campaign.sh --execute --resume" in result.stdout
    assert "cleanup commands (not executed):" in result.stdout
    assert (
        f"gcloud artifacts repositories delete qldpc-fno-{CAMPAIGN_ID}"
        in result.stdout
    )
    assert "remove-iam-policy-binding" not in result.stdout
    assert "gcloud projects add-iam-policy-binding" not in result.stdout
    assert "canonical cloud execution is blocked" in result.stdout


def test_dry_run_falls_back_to_verified_compute_service_account_when_command_is_absent(
    tmp_path: Path,
) -> None:
    result, calls = _launch(
        tmp_path,
        environment_overrides={"FAKE_DEFAULT_BUILD_SA_UNSUPPORTED": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert f"build_service_account={BUILD_SERVICE_ACCOUNT}" in result.stdout
    assert calls[:4] == [
        ["config", "get-value", "project"],
        [
            "builds",
            "get-default-service-account",
            "--project=science-project",
            "--format=value(serviceAccountEmail)",
        ],
        ["projects", "describe", "science-project", "--format=value(projectNumber)"],
        [
            "iam",
            "service-accounts",
            "describe",
            BUILD_SERVICE_ACCOUNT,
            "--project=science-project",
            "--format=value(email)",
        ],
    ]


def test_preferred_command_accepts_google_managed_legacy_build_account(
    tmp_path: Path,
) -> None:
    legacy_account = f"{PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"
    result, calls = _launch(
        tmp_path,
        "--execute",
        "--reduced",
        environment_overrides={
            "FAKE_DEFAULT_BUILD_SA": (
                f"projects/science-project/serviceAccounts/{legacy_account}"
            ),
            "FAKE_BUILD_SA_DESCRIBE_FAILURE": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert f"build_service_account={legacy_account}" in result.stdout
    assert not any(
        call[:3] == ["iam", "service-accounts", "describe"]
        and call[3] == legacy_account
        for call in calls
    )
    repository_writer = next(
        call
        for call in calls
        if call[:3] == ["artifacts", "repositories", "add-iam-policy-binding"]
    )
    assert f"--member=serviceAccount:{legacy_account}" in repository_writer


@pytest.mark.parametrize(
    "environment_overrides",
    [
        {"FAKE_DEFAULT_BUILD_SA_FAILURE": "1"},
        {"FAKE_DEFAULT_BUILD_SA": "attacker@example.com"},
        {
            "FAKE_DEFAULT_BUILD_SA": (
                "999999999999-compute@developer.gserviceaccount.com"
            )
        },
        {"FAKE_DESCRIBED_BUILD_SA": "different@developer.gserviceaccount.com"},
        {"FAKE_BUILD_SA_DESCRIBE_FAILURE": "1"},
        {"FAKE_PROJECT_NUMBER_FAILURE": "1"},
        {
            "FAKE_DEFAULT_BUILD_SA_UNSUPPORTED": "1",
            "FAKE_PROJECT_NUMBER": "not-a-project-number",
        },
    ],
)
def test_build_service_account_resolution_fails_closed_before_mutation(
    tmp_path: Path,
    environment_overrides: dict[str, str],
) -> None:
    result, calls = _launch(
        tmp_path,
        "--execute",
        "--reduced",
        environment_overrides=environment_overrides,
    )

    assert result.returncode == 2
    assert "Cloud Build service account" in result.stderr
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


def test_reduced_execute_creates_one_bounded_cpu_job(tmp_path: Path) -> None:
    result, calls = _launch(tmp_path, "--execute", "--reduced")
    expected_pinned_image = (
        f"us-central1-docker.pkg.dev/science-project/qldpc-fno-{CAMPAIGN_ID}"
        f"/accuracy-campaign@{DIGEST}"
    )

    assert result.returncode == 0, result.stderr
    assert calls[0] == ["config", "get-value", "project"]
    mutation_calls = [
        call
        for call in calls
        if "describe" not in call and call != ["config", "get-value", "project"]
        and call[:2] != ["builds", "get-default-service-account"]
    ]
    assert [call[:3] for call in mutation_calls] == [
        ["artifacts", "repositories", "create"],
        ["artifacts", "repositories", "add-iam-policy-binding"],
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
    assert "--parallelism=1" in create_job
    assert not any(argument.startswith("--execution-environment") for argument in create_job)
    assert f"--image={expected_pinned_image}" in create_job
    assert not any(argument.startswith("--gpu") for argument in create_job)
    repository_writer = next(
        call
        for call in calls
        if call[:3] == ["artifacts", "repositories", "add-iam-policy-binding"]
    )
    assert repository_writer[3] == f"qldpc-fno-{CAMPAIGN_ID}"
    assert f"--member=serviceAccount:{BUILD_SERVICE_ACCOUNT}" in repository_writer
    assert "--role=roles/artifactregistry.writer" in repository_writer
    assert "--location=us-central1" in repository_writer
    assert "--project=science-project" in repository_writer
    assert not any(
        call[:2] == ["projects", "add-iam-policy-binding"] for call in calls
    )
    assert not any(
        call[:3] == ["artifacts", "repositories", "add-iam-policy-binding"]
        and any(
            argument
            == "--member=serviceAccount:"
            "qfno-abcdef0123456789abcdef01@science-project.iam.gserviceaccount.com"
            for argument in call
        )
        for call in calls
    )
    env_argument = next(argument for argument in create_job if argument.startswith("--set-env-vars="))
    assert f"CAMPAIGN_BUCKET=science-project-{CAMPAIGN_ID}" in env_argument
    assert f"CAMPAIGN_PREFIX=campaigns/{CAMPAIGN_ID}/{COMMIT}" in env_argument
    assert f"CAMPAIGN_GIT_COMMIT={COMMIT}" in env_argument
    assert "CAMPAIGN_BOOTSTRAP_SAMPLES" not in env_argument
    assert "CAMPAIGN_CONFIG=/app/configs/accuracy_campaign_cloud_reduced.json" in env_argument
    assert f"CAMPAIGN_IMAGE={expected_pinned_image}" in env_argument
    assert f"CAMPAIGN_IMAGE_DIGEST={DIGEST}" in env_argument
    assert "CAMPAIGN_OUTER_TIMEOUT_SECONDS=28800" in env_argument
    assert "CAMPAIGN_WORK_CUTOFF_SECONDS=26100" in env_argument
    assert "CAMPAIGN_FINALIZATION_RESERVE_SECONDS=2700" in env_argument
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
    assert "--wait" in execute_job
    assert "--async" not in execute_job
    assert not any("delete" in call for call in calls)


def test_canonical_execute_requires_multi_execution_acknowledgement(tmp_path: Path) -> None:
    result, calls = _launch(tmp_path, "--execute")

    assert result.returncode == 2
    assert "--multi-execution" in result.stderr
    assert not any("create" in call or call[:2] == ["builds", "submit"] for call in calls)


def test_canonical_execute_fails_closed_at_representative_benchmark_gate(
    tmp_path: Path,
) -> None:
    result, calls = _launch(tmp_path, "--execute", "--multi-execution")

    assert result.returncode == 2
    assert "representative decoder benchmark gate" in result.stderr
    assert calls == [
        ["config", "get-value", "project"],
        [
            "builds",
            "get-default-service-account",
            "--project=science-project",
            "--format=value(serviceAccountEmail)",
        ],
        ["projects", "describe", "science-project", "--format=value(projectNumber)"],
        [
            "iam",
            "service-accounts",
            "describe",
            BUILD_SERVICE_ACCOUNT,
            "--project=science-project",
            "--format=value(email)",
        ],
    ]


def test_resume_verifies_exact_job_identity_and_never_recreates_resources(tmp_path: Path) -> None:
    result, calls = _launch(
        tmp_path,
        "--execute",
        "--resume",
        environment_overrides={"FAKE_GCLOUD_RESUME": "1"},
    )

    assert result.returncode == 2
    assert "mode=resume" in result.stdout
    assert "representative decoder benchmark gate" in result.stderr
    mutation_calls = [
        call
        for call in calls
        if "describe" not in call and call != ["config", "get-value", "project"]
        and call[:2] != ["builds", "get-default-service-account"]
        and call[:2] != ["storage", "cp"]
    ]
    assert mutation_calls == []
    assert not any("create" in call or call[:2] == ["builds", "submit"] for call in calls)

    mismatched, mismatch_calls = _launch(
        tmp_path / "mismatch",
        "--execute",
        "--resume",
        environment_overrides={
            "FAKE_GCLOUD_RESUME": "1",
            "FAKE_JOB_IDENTITY": "wrong-identity",
        },
    )
    assert mismatched.returncode == 2
    assert "identity" in mismatched.stderr
    assert not any(call[:3] == ["run", "jobs", "execute"] for call in mismatch_calls)


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"FAKE_JOB_STORE": "gs://wrong/store"}, "environment"),
        (
            {"FAKE_JOB_SERVICE_ACCOUNT": "wrong@science-project.iam.gserviceaccount.com"},
            "service_account",
        ),
        ({"FAKE_JOB_TIMEOUT_SECONDS": "28799"}, "timeout_seconds"),
    ],
)
def test_resume_rejects_any_meaningful_cloud_job_contract_drift(
    tmp_path: Path,
    environment: dict[str, str],
    message: str,
) -> None:
    result, calls = _launch(
        tmp_path,
        "--execute",
        "--resume",
        environment_overrides={"FAKE_GCLOUD_RESUME": "1", **environment},
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert not any(call[:3] == ["run", "jobs", "execute"] for call in calls)


def test_reduced_execute_uses_non_scientific_config_and_waits(tmp_path: Path) -> None:
    result, calls = _launch(tmp_path, "--execute", "--reduced")

    assert result.returncode == 0, result.stderr
    assert "campaign_mode=reduced_non_scientific" in result.stdout
    create_job = next(call for call in calls if call[:3] == ["run", "jobs", "create"])
    env_argument = next(argument for argument in create_job if argument.startswith("--set-env-vars="))
    assert "CAMPAIGN_CONFIG=/app/configs/accuracy_campaign_cloud_reduced.json" in env_argument
    assert "--args=--campaign-mode=reduced_non_scientific,--calibration-grid-limit=1" in create_job
    assert not any("bootstrap-samples" in argument for argument in create_job)
    execute_job = next(call for call in calls if call[:3] == ["run", "jobs", "execute"])
    assert "--wait" in execute_job
    assert "--async" not in execute_job


def test_cloud_build_archive_excludes_ignored_worktree_sentinel(tmp_path: Path) -> None:
    sentinel = Path("src/qldpc_fno/ignored-cloud-context-sentinel.secret")
    sentinel.write_text("must never upload")
    try:
        result, calls = _launch(tmp_path, "--execute", "--reduced")
    finally:
        sentinel.unlink(missing_ok=True)

    assert result.returncode == 0, result.stderr
    build = next(call for call in calls if call[:2] == ["builds", "submit"])
    assert build[-1].endswith(".tar.gz")
    members = (tmp_path / "archive-members.txt").read_text().splitlines()
    assert "src/qldpc_fno/ignored-cloud-context-sentinel.secret" not in members
    assert "Dockerfile" in members
    assert "src/qldpc_fno/campaign/runner.py" in members


@pytest.mark.parametrize("existing", ["repository", "bucket", "job", "service-account"])
def test_execute_refuses_existing_campaign_resources_before_mutation(
    tmp_path: Path, existing: str
) -> None:
    result, calls = _launch(
        tmp_path,
        "--execute",
        "--reduced",
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
        "--reduced",
        environment_overrides={"FAKE_GCLOUD_DESCRIBE_ERROR": "1"},
    )

    assert result.returncode == 2
    assert "cannot verify campaign repository absence" in result.stderr
    assert not any("create" in call or call[:2] == ["builds", "submit"] for call in calls)


def test_repository_writer_grant_failure_prevents_build_and_later_mutations(
    tmp_path: Path,
) -> None:
    result, calls = _launch(
        tmp_path,
        "--execute",
        "--reduced",
        environment_overrides={"FAKE_BUILD_WRITER_GRANT_FAILURE": "1"},
    )

    assert result.returncode == 2
    grant_index = next(
        index
        for index, call in enumerate(calls)
        if call[:3] == ["artifacts", "repositories", "add-iam-policy-binding"]
    )
    assert calls[grant_index - 1][:3] == ["artifacts", "repositories", "create"]
    assert calls[grant_index + 1 :] == []


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
    assert ignored[0] == "**"
    assert "!Dockerfile" in ignored
    assert "!src/qldpc_fno/**/*.py" in ignored
    assert "!experiments/16_calibrate_hybrid_priors.py" in ignored
    assert "COPY scripts ./scripts" not in dockerfile
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
