from __future__ import annotations

import copy

import pytest

from qldpc_fno.campaign.cloud_contract import verify_cloud_job_contract


def _contract() -> tuple[dict[str, object], dict[str, object]]:
    env = {
        "CAMPAIGN_BOOTSTRAP_SAMPLES": "10000",
        "CAMPAIGN_BUCKET": "science-project-accuracy-a1b2",
        "CAMPAIGN_CALIBRATION_GRID_LIMIT": "",
        "CAMPAIGN_CANONICAL_CONFIG": "/app/configs/accuracy_campaign.json",
        "CAMPAIGN_CLOUD_JOB": "qldpc-fno-accuracy-a1b2",
        "CAMPAIGN_CLOUD_PROJECT": "science-project",
        "CAMPAIGN_CLOUD_REGION": "us-central1",
        "CAMPAIGN_CODE": "/app/campaign-code",
        "CAMPAIGN_CONFIG": "/app/configs/accuracy_campaign.json",
        "CAMPAIGN_GIT_COMMIT": "0" * 40,
        "CAMPAIGN_IMAGE_DIGEST": "sha256:" + "1" * 64,
        "CAMPAIGN_MODE": "canonical",
        "CAMPAIGN_PREFIX": "campaigns/accuracy-a1b2/" + "0" * 40,
        "CAMPAIGN_STORE": (
            "gs://science-project-accuracy-a1b2/campaigns/accuracy-a1b2/" + "0" * 40
        ),
        "CAMPAIGN_WORKDIR": "/tmp/qldpc-fno-work",
    }
    expected: dict[str, object] = {
        "args": [],
        "command": [],
        "cpu": "8",
        "env": env,
        "execution_environment": "gen2",
        "identity_label": "abc123",
        "image": "image@sha256:" + "1" * 64,
        "memory": "32Gi",
        "mode_label": "canonical",
        "parallelism": 1,
        "service_account": "campaign@science-project.iam.gserviceaccount.com",
        "task_count": 1,
        "timeout_seconds": 28_800,
        "max_retries": 0,
    }
    payload: dict[str, object] = {
        "metadata": {
            "labels": {
                "qldpc-fno-identity": expected["identity_label"],
                "qldpc-fno-mode": expected["mode_label"],
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
                                    "args": [],
                                    "command": [],
                                    "env": [
                                        {"name": name, "value": value}
                                        for name, value in sorted(env.items())
                                    ],
                                    "image": expected["image"],
                                    "resources": {
                                        "limits": {"cpu": "8", "memory": "32Gi"}
                                    },
                                }
                            ],
                            "executionEnvironment": "gen2",
                            "maxRetries": 0,
                            "serviceAccountName": expected["service_account"],
                            "timeoutSeconds": "28800",
                        }
                    },
                }
            }
        },
    }
    return payload, expected


def test_cloud_job_contract_accepts_only_exact_runtime_identity() -> None:
    payload, expected = _contract()

    verify_cloud_job_contract(payload, expected)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("spec", "template", "spec", "taskCount"), 2),
        (
            (
                "spec",
                "template",
                "spec",
                "template",
                "spec",
                "containers",
                0,
                "image",
            ),
            "image@sha256:" + "2" * 64,
        ),
        (
            (
                "spec",
                "template",
                "spec",
                "template",
                "spec",
                "serviceAccountName",
            ),
            "wrong@science-project.iam.gserviceaccount.com",
        ),
        (
            (
                "spec",
                "template",
                "spec",
                "template",
                "spec",
                "timeoutSeconds",
            ),
            "3600",
        ),
    ],
)
def test_cloud_job_contract_rejects_drifted_fields(
    path: tuple[object, ...], replacement: object
) -> None:
    payload, expected = _contract()
    current: object = payload
    for part in path[:-1]:
        current = current[part]  # type: ignore[index]
    current[path[-1]] = replacement  # type: ignore[index]

    with pytest.raises(ValueError, match="Cloud Run job contract"):
        verify_cloud_job_contract(payload, expected)


def test_cloud_job_contract_rejects_extra_or_missing_environment() -> None:
    payload, expected = _contract()
    extra = copy.deepcopy(payload)
    containers = extra["spec"]["template"]["spec"]["template"]["spec"]["containers"]
    containers[0]["env"].append({"name": "UNBOUND_CONTROL", "value": "1"})
    with pytest.raises(ValueError, match="environment"):
        verify_cloud_job_contract(extra, expected)

    missing = copy.deepcopy(payload)
    containers = missing["spec"]["template"]["spec"]["template"]["spec"]["containers"]
    containers[0]["env"].pop()
    with pytest.raises(ValueError, match="environment"):
        verify_cloud_job_contract(missing, expected)
