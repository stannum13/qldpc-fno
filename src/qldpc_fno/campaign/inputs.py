"""Shared immutable input bootstrap for local and Cloud campaign executions."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.campaign.config import CampaignConfig
from qldpc_fno.campaign.storage import ArtifactStore, LocalArtifactStore, materialize_completion
from qldpc_fno.training.calibration import CALIBRATION_GRID

_STAGE_PREFIXES = ("pilot", "shards", "training", "calibration", "evaluation", "summary")
_CLOUD_IDENTITY_FIELDS = {
    "bucket",
    "finalization_reserve_seconds",
    "image",
    "image_digest",
    "job",
    "kind",
    "outer_timeout_seconds",
    "prefix",
    "project",
    "region",
    "service_account",
    "store",
    "work_cutoff_seconds",
}
_LOCAL_IDENTITY_FIELDS = {"kind", "store"}
_TRUSTED_CANONICAL_CONFIG_SHA256 = {
    "accuracy_campaign.json": "334f8087ff86dd51fa34551e663ea65e7650819fca3552b704f3174d78d5ff78",
    "accuracy_disconfirm_p0375.json": (
        "8df94770eef8207431935d30c30b3ab8f8243ace47b1ae8401452f6803714717"
    ),
}
_RUN_MODE_FIELDS = {
    "canonical_config",
    "canonical_config_sha256",
    "code_manifest_sha256",
    "effective_config_sha256",
    "execution_controls",
    "execution_identity",
    "git_commit",
    "mode",
    "overrides",
    "schema_version",
    "scientific_claims_permitted",
}


@dataclass(frozen=True, slots=True)
class CampaignInputRequest:
    """Expected identity for one immutable campaign store."""

    canonical_config: Path
    effective_config: Path
    code: Path
    git_commit: str
    campaign_mode: str
    calibration_grid_limit: int | None
    execution_identity: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PreparedCampaignInputs:
    """Materialized and reverified campaign input paths."""

    config: Path
    code: Path
    run_mode: Path


def _validate_trusted_canonical_config(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("trusted committed canonical config is unavailable")
    expected_sha256 = _TRUSTED_CANONICAL_CONFIG_SHA256.get(path.name)
    if expected_sha256 is None or sha256_file(path) != expected_sha256:
        raise ValueError("canonical config does not match trusted committed canonical config")


def _validate_execution_identity(
    identity: Mapping[str, object],
    *,
    config: CampaignConfig,
) -> None:
    if not isinstance(identity, Mapping) or identity.get("kind") not in {"local", "cloud"}:
        raise ValueError("campaign execution identity is malformed")
    if not isinstance(identity.get("store"), str) or not identity["store"]:
        raise ValueError("campaign execution identity store is missing")
    if identity["kind"] == "local":
        if set(identity) != _LOCAL_IDENTITY_FIELDS:
            raise ValueError("local campaign execution identity fields are incomplete")
        return
    if set(identity) != _CLOUD_IDENTITY_FIELDS:
        raise ValueError("cloud campaign execution identity fields are incomplete")
    string_fields = _CLOUD_IDENTITY_FIELDS - {
        "finalization_reserve_seconds",
        "outer_timeout_seconds",
        "work_cutoff_seconds",
    }
    if any(not isinstance(identity.get(field), str) or not identity[field] for field in string_fields):
        raise ValueError("cloud campaign execution identity string fields are malformed")
    digest = identity["image_digest"]
    image = identity["image"]
    if (
        re.fullmatch(r"sha256:[0-9a-f]{64}", str(digest)) is None
        or not str(image).endswith(f"@{digest}")
    ):
        raise ValueError("cloud campaign execution identity image digest is malformed")
    expected_numeric = {
        "finalization_reserve_seconds": config.checkpoint_grace_seconds,
        "outer_timeout_seconds": config.cloud_timeout_seconds,
        "work_cutoff_seconds": config.cloud_timeout_seconds - config.checkpoint_grace_seconds,
    }
    if any(
        type(identity.get(field)) is not int or identity[field] != value
        for field, value in expected_numeric.items()
    ):
        raise ValueError("cloud campaign execution identity deadline controls are malformed")
    if identity["store"] != f"gs://{identity['bucket']}/{identity['prefix']}":
        raise ValueError("cloud campaign execution identity store is inconsistent")


def _validate_request(request: CampaignInputRequest) -> tuple[CampaignConfig, CampaignConfig]:
    _validate_trusted_canonical_config(request.canonical_config)
    canonical = CampaignConfig.from_json(request.canonical_config)
    effective = CampaignConfig.from_json(request.effective_config)
    if re.fullmatch(r"[0-9a-f]{40}", request.git_commit) is None:
        raise ValueError("campaign input Git commit must be a full lowercase SHA-1")
    if request.campaign_mode == "canonical":
        if request.calibration_grid_limit is not None:
            raise ValueError("canonical campaign controls cannot reduce calibration")
        if request.canonical_config.read_bytes() != request.effective_config.read_bytes():
            raise ValueError("canonical campaign controls require the committed canonical config")
    elif request.campaign_mode == "reduced_non_scientific":
        if (
            type(request.calibration_grid_limit) is not int
            or not 0 < request.calibration_grid_limit <= len(CALIBRATION_GRID)
        ):
            raise ValueError("reduced campaign requires an explicit bounded calibration grid")
    else:
        raise ValueError("campaign input mode is invalid")
    _validate_execution_identity(request.execution_identity, config=effective)
    return canonical, effective


def _overrides(canonical: CampaignConfig, effective: CampaignConfig) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for field in sorted(CampaignConfig._FIELD_NAMES):
        value = getattr(effective, field)
        if value != getattr(canonical, field):
            overrides[field] = list(value) if isinstance(value, tuple) else value
    return overrides


def _type_strict_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(  # type: ignore[arg-type]
            _type_strict_equal(actual[key], value)  # type: ignore[index]
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(  # type: ignore[arg-type]
            _type_strict_equal(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected, strict=True)  # type: ignore[arg-type]
        )
    return bool(actual == expected)


def _expected_mode(
    request: CampaignInputRequest,
    canonical: CampaignConfig,
    effective: CampaignConfig,
) -> dict[str, object]:
    code_manifest = request.code / "code.json"
    if not code_manifest.is_file() or code_manifest.is_symlink():
        raise ValueError("campaign code manifest is unavailable")
    return {
        "canonical_config": request.canonical_config.name,
        "canonical_config_sha256": sha256_file(request.canonical_config),
        "code_manifest_sha256": sha256_file(code_manifest),
        "effective_config_sha256": sha256_file(request.effective_config),
        "execution_controls": {"calibration_grid_limit": request.calibration_grid_limit},
        "execution_identity": dict(request.execution_identity),
        "git_commit": request.git_commit,
        "mode": request.campaign_mode,
        "overrides": _overrides(canonical, effective),
        "schema_version": 3,
        "scientific_claims_permitted": request.campaign_mode == "canonical",
    }


def verify_campaign_run_mode(
    path: Path,
    *,
    canonical_config_path: Path | None = None,
    config_path: Path,
    code_manifest_path: Path,
    git_commit: str,
) -> dict[str, object]:
    """Verify immutable run mode, claim policy, and input provenance."""
    if path.is_symlink():
        raise ValueError("campaign run-mode manifest must not be a symlink")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("campaign run-mode manifest is unreadable") from error
    if not isinstance(payload, dict) or set(payload) != _RUN_MODE_FIELDS:
        raise ValueError("campaign run-mode manifest schema is malformed")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 3:
        raise ValueError("campaign run-mode schema version is unsupported")

    mode = payload.get("mode")
    claims_permitted = payload.get("scientific_claims_permitted")
    if (
        mode not in {"canonical", "reduced_non_scientific"}
        or type(claims_permitted) is not bool
        or claims_permitted is not (mode == "canonical")
    ):
        raise ValueError("campaign run mode and claim policy are inconsistent")
    if payload.get("effective_config_sha256") != sha256_file(config_path):
        raise ValueError("campaign run-mode configuration provenance mismatch")
    if payload.get("code_manifest_sha256") != sha256_file(code_manifest_path):
        raise ValueError("campaign run-mode code provenance mismatch")
    if payload.get("git_commit") != git_commit:
        raise ValueError("campaign run-mode Git provenance mismatch")

    canonical_name = payload.get("canonical_config")
    canonical_digest = payload.get("canonical_config_sha256")
    overrides = payload.get("overrides")
    controls = payload.get("execution_controls")
    identity = payload.get("execution_identity")
    if (
        not isinstance(canonical_name, str)
        or not canonical_name
        or Path(canonical_name).name != canonical_name
        or not isinstance(canonical_digest, str)
        or not isinstance(overrides, dict)
        or not isinstance(controls, dict)
        or set(controls) != {"calibration_grid_limit"}
        or not isinstance(identity, dict)
    ):
        raise ValueError("campaign run-mode policy fields are malformed")
    if canonical_config_path is None:
        canonical_config_path = Path(__file__).resolve().parents[3] / "configs" / canonical_name
    request = CampaignInputRequest(
        canonical_config=canonical_config_path,
        effective_config=config_path,
        code=code_manifest_path.parent,
        git_commit=git_commit,
        campaign_mode=str(mode),
        calibration_grid_limit=controls["calibration_grid_limit"],
        execution_identity=identity,
    )
    canonical, effective = _validate_request(request)
    expected = _expected_mode(request, canonical, effective)
    if not _type_strict_equal(payload, expected):
        raise ValueError("campaign run-mode policy does not match trusted inputs")
    return expected


def _stage_evidence_exists(
    store: ArtifactStore,
    deadline_monotonic: float | None,
) -> bool:
    return any(
        store.exists(
            f"{prefix}/_COMPLETE.json",
            deadline_monotonic=deadline_monotonic,
        )
        or store.exists(
            f".checkpoints/{prefix}/00000000/_COMPLETE.json",
            deadline_monotonic=deadline_monotonic,
        )
        for prefix in _STAGE_PREFIXES
    )


def prepare_campaign_inputs(
    store: ArtifactStore,
    workspace: Path,
    request: CampaignInputRequest,
    *,
    deadline_monotonic: float | None = None,
) -> PreparedCampaignInputs:
    """Publish or verify inputs, materialize them, and enforce exact run identity."""
    canonical, effective = _validate_request(request)
    expected_mode = _expected_mode(request, canonical, effective)
    verified = store.verify_completion("inputs", deadline_monotonic=deadline_monotonic)
    if not verified:
        try:
            published_input_evidence = store.exists(
                "inputs/_COMPLETE.json",
                deadline_monotonic=deadline_monotonic,
            )
            stage_evidence = _stage_evidence_exists(store, deadline_monotonic)
        except ValueError as error:
            raise ValueError("established campaign input publication is corrupt") from error
        if published_input_evidence or stage_evidence:
            raise ValueError("established campaign input publication is corrupt")
        with tempfile.TemporaryDirectory(prefix="qldpc-fno-inputs-") as temporary:
            staging = Path(temporary) / "inputs"
            staging.mkdir()
            shutil.copyfile(request.effective_config, staging / "config.json")
            shutil.copytree(request.code, staging / "code")
            write_canonical_json(staging / "run-mode.json", expected_mode)
            store.publish_directory(
                staging,
                "inputs",
                deadline_monotonic=deadline_monotonic,
            )

    materialized = workspace / "inputs"
    materialize_completion(
        store,
        "inputs",
        materialized,
        deadline_monotonic=deadline_monotonic,
    )
    run_mode_path = materialized / "run-mode.json"
    actual_mode = json.loads(run_mode_path.read_text())
    if not _type_strict_equal(actual_mode, expected_mode):
        raise ValueError("campaign input identity does not match the immutable store")
    config_path = materialized / "config.json"
    code_path = materialized / "code"
    if (
        sha256_file(config_path) != expected_mode["effective_config_sha256"]
        or sha256_file(code_path / "code.json") != expected_mode["code_manifest_sha256"]
    ):
        raise ValueError("campaign input materialization failed provenance verification")
    CampaignConfig.from_json(config_path)
    return PreparedCampaignInputs(config=config_path, code=code_path, run_mode=run_mode_path)


def verify_downloaded_cloud_inputs(
    store_root: Path,
    canonical_config: Path,
    *,
    git_commit: str,
    campaign_mode: str,
    calibration_grid_limit: int | None,
    expected_execution_identity: Mapping[str, object],
) -> dict[str, object]:
    """Verify a recursively downloaded Cloud ``inputs/`` publication for resume."""
    store = LocalArtifactStore(store_root)
    if not store.verify_completion("inputs"):
        raise ValueError("downloaded cloud campaign inputs are not a verified publication")
    with tempfile.TemporaryDirectory(prefix="qldpc-fno-cloud-input-verify-") as temporary:
        materialized = Path(temporary) / "inputs"
        materialize_completion(store, "inputs", materialized)
        mode_path = materialized / "run-mode.json"
        mode = json.loads(mode_path.read_text())
        if not isinstance(mode, dict) or not isinstance(mode.get("execution_identity"), dict):
            raise TypeError("downloaded cloud campaign run-mode identity is malformed")
        identity = mode["execution_identity"]
        for key, value in expected_execution_identity.items():
            if identity.get(key) != value:
                raise ValueError(f"downloaded cloud campaign input identity differs in {key}")
        digest = identity.get("image_digest")
        image = identity.get("image")
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
            or not isinstance(image, str)
            or not image.endswith(f"@{digest}")
        ):
            raise ValueError("downloaded cloud campaign image digest identity is malformed")
        request = CampaignInputRequest(
            canonical_config=canonical_config,
            effective_config=materialized / "config.json",
            code=materialized / "code",
            git_commit=git_commit,
            campaign_mode=campaign_mode,
            calibration_grid_limit=calibration_grid_limit,
            execution_identity=identity,
        )
        canonical, effective = _validate_request(request)
        if not _type_strict_equal(mode, _expected_mode(request, canonical, effective)):
            raise ValueError("downloaded cloud campaign run-mode manifest is inconsistent")
        return dict(identity)
