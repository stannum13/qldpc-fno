"""Strict normalization for the immutable Cloud Run campaign job contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"Cloud Run job contract has malformed {label}")
    return value


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TypeError(f"Cloud Run job contract has malformed {label}")
    return value


def _nested(payload: Mapping[str, object], *keys: str) -> object:
    current: object = payload
    for key in keys:
        current = _mapping(current, label="specification").get(key)
    return current


def _timeout_seconds(value: object) -> int:
    if type(value) is int:
        return value
    if isinstance(value, str):
        normalized = value.removesuffix("s")
        if normalized.isdecimal():
            return int(normalized)
    raise ValueError("Cloud Run job contract has malformed timeout")


def _environment(container: Mapping[str, object]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in _sequence(container.get("env", []), label="environment"):
        entry = _mapping(raw, label="environment entry")
        if set(entry) != {"name", "value"}:
            raise ValueError("Cloud Run job contract environment is not literal and exact")
        name = entry.get("name")
        value = entry.get("value")
        if not isinstance(name, str) or not isinstance(value, str) or name in result:
            raise ValueError("Cloud Run job contract environment is malformed or duplicated")
        result[name] = value
    return result


def verify_cloud_job_contract(
    payload: object,
    expected: Mapping[str, object],
) -> None:
    """Reject any drift in meaningful Cloud Run job execution fields."""
    job = _mapping(payload, label="document")
    labels = _mapping(_nested(job, "metadata", "labels"), label="labels")
    campaign_labels = {
        key: value for key, value in labels.items() if key.startswith("qldpc-fno-")
    }
    expected_labels = {
        "qldpc-fno-identity": expected.get("identity_label"),
        "qldpc-fno-mode": expected.get("mode_label"),
    }
    if campaign_labels != expected_labels:
        raise ValueError("Cloud Run job contract identity labels differ")

    execution = _mapping(
        _nested(job, "spec", "template", "spec"),
        label="execution template",
    )
    task = _mapping(
        _nested(job, "spec", "template", "spec", "template", "spec"),
        label="task template",
    )
    containers = _sequence(task.get("containers"), label="containers")
    if len(containers) != 1:
        raise ValueError("Cloud Run job contract must contain exactly one container")
    container = _mapping(containers[0], label="container")
    unexpected_container_fields = set(container) - {
        "args",
        "command",
        "env",
        "image",
        "name",
        "resources",
    }
    if unexpected_container_fields:
        raise ValueError("Cloud Run job contract contains unbound container fields")
    resources = _mapping(container.get("resources"), label="resources")
    if set(resources) - {"limits"}:
        raise ValueError("Cloud Run job contract contains unbound resource fields")
    limits = _mapping(resources.get("limits"), label="resource limits")
    declared_execution_environment = task.get("executionEnvironment")
    if declared_execution_environment not in (None, "gen2"):
        raise ValueError("Cloud Run job contract execution environment differs")
    actual = {
        "args": list(_sequence(container.get("args", []), label="arguments")),
        "command": list(_sequence(container.get("command", []), label="command")),
        "cpu": limits.get("cpu"),
        "env": _environment(container),
        # Cloud Run jobs are a generation-2-only product. Some API projections
        # omit the non-configurable field; reject any contradictory declaration.
        "execution_environment": "gen2",
        "image": container.get("image"),
        "max_retries": task.get("maxRetries"),
        "memory": limits.get("memory"),
        "parallelism": execution.get("parallelism"),
        "service_account": task.get("serviceAccountName"),
        "task_count": execution.get("taskCount"),
        "timeout_seconds": _timeout_seconds(task.get("timeoutSeconds")),
    }
    expected_runtime = {
        key: expected.get(key)
        for key in (
            "args",
            "command",
            "cpu",
            "env",
            "execution_environment",
            "image",
            "max_retries",
            "memory",
            "parallelism",
            "service_account",
            "task_count",
            "timeout_seconds",
        )
    }
    if actual.get("env") != expected_runtime.get("env"):
        raise ValueError("Cloud Run job contract environment differs")
    if actual != expected_runtime:
        differing = sorted(key for key in actual if actual[key] != expected_runtime[key])
        raise ValueError(f"Cloud Run job contract differs in fields: {differing}")
    for field in ("volumes", "vpcAccess", "nodeSelector"):
        if task.get(field) not in (None, [], {}):
            raise ValueError(f"Cloud Run job contract contains unbound {field}")
