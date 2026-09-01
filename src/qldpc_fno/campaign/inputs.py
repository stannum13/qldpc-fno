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


@dataclass(frozen=True, slots=True)
class CampaignInputRequest:
    """Expected identity for one immutable campaign store."""

    canonical_config: Path
    effective_config: Path
    code: Path
    git_commit: str
    campaign_mode: str
    calibration_grid_limit: int | None
    bootstrap_samples: int
    execution_identity: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PreparedCampaignInputs:
    """Materialized and reverified campaign input paths."""

    config: Path
    code: Path
    run_mode: Path


def _validate_request(request: CampaignInputRequest) -> tuple[CampaignConfig, CampaignConfig]:
    canonical = CampaignConfig.from_json(request.canonical_config)
    effective = CampaignConfig.from_json(request.effective_config)
    if re.fullmatch(r"[0-9a-f]{40}", request.git_commit) is None:
        raise ValueError("campaign input Git commit must be a full lowercase SHA-1")
    if type(request.bootstrap_samples) is not int or request.bootstrap_samples <= 0:
        raise ValueError("campaign bootstrap samples must be a positive integer")
    if request.campaign_mode == "canonical":
        if request.calibration_grid_limit is not None or request.bootstrap_samples != 10_000:
            raise ValueError("canonical campaign controls cannot reduce calibration or bootstrap")
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
    identity = request.execution_identity
    if not isinstance(identity, Mapping) or identity.get("kind") not in {"local", "cloud"}:
        raise ValueError("campaign execution identity is malformed")
    if not isinstance(identity.get("store"), str) or not identity["store"]:
        raise ValueError("campaign execution identity store is missing")
    if identity["kind"] == "cloud" and set(identity) != _CLOUD_IDENTITY_FIELDS:
        raise ValueError("cloud campaign execution identity fields are incomplete")
    return canonical, effective


def _overrides(canonical: CampaignConfig, effective: CampaignConfig) -> dict[str, object]:
    return {
        field: getattr(effective, field)
        for field in sorted(CampaignConfig._FIELD_NAMES)
        if getattr(effective, field) != getattr(canonical, field)
    }


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
        "execution_controls": {
            "bootstrap_samples": request.bootstrap_samples,
            "calibration_grid_limit": request.calibration_grid_limit,
        },
        "execution_identity": dict(request.execution_identity),
        "git_commit": request.git_commit,
        "mode": request.campaign_mode,
        "overrides": _overrides(canonical, effective),
        "schema_version": 2,
        "scientific_claims_permitted": request.campaign_mode == "canonical",
    }


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
    if actual_mode != expected_mode:
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
    bootstrap_samples: int,
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
            bootstrap_samples=bootstrap_samples,
            execution_identity=identity,
        )
        canonical, effective = _validate_request(request)
        if mode != _expected_mode(request, canonical, effective):
            raise ValueError("downloaded cloud campaign run-mode manifest is inconsistent")
        return dict(identity)
