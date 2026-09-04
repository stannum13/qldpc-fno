from __future__ import annotations

import json
from pathlib import Path

import pytest

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.campaign.inputs import (
    CampaignInputRequest,
    prepare_campaign_inputs,
    verify_campaign_run_mode,
)
from qldpc_fno.campaign.storage import LocalArtifactStore


def _request(tmp_path: Path, **overrides: object) -> CampaignInputRequest:
    code = tmp_path / "code"
    code.mkdir(exist_ok=True)
    (code / "code.json").write_text('{"name":"lp_3_7_16"}\n')
    values: dict[str, object] = {
        "calibration_grid_limit": None,
        "campaign_mode": "canonical",
        "canonical_config": Path("configs/accuracy_campaign.json"),
        "code": code,
        "effective_config": Path("configs/accuracy_campaign.json"),
        "execution_identity": {
            "kind": "local",
            "store": str((tmp_path / "store").resolve()),
        },
        "git_commit": "a" * 40,
    }
    values.update(overrides)
    return CampaignInputRequest(**values)  # type: ignore[arg-type]


def _cloud_identity() -> dict[str, object]:
    digest = f"sha256:{'b' * 64}"
    return {
        "bucket": "campaign-bucket",
        "finalization_reserve_seconds": 2700,
        "image": f"us-central1-docker.pkg.dev/project/repository/image@{digest}",
        "image_digest": digest,
        "job": "qldpc-fno-campaign",
        "kind": "cloud",
        "outer_timeout_seconds": 28800,
        "prefix": "campaigns/campaign/commit",
        "project": "project",
        "region": "us-central1",
        "service_account": "campaign@project.iam.gserviceaccount.com",
        "store": "gs://campaign-bucket/campaigns/campaign/commit",
        "work_cutoff_seconds": 26100,
    }


def test_shared_input_bootstrap_publishes_materializes_and_revalidates(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    request = _request(tmp_path)

    prepared = prepare_campaign_inputs(store, tmp_path / "work", request)

    assert store.verify_completion("inputs") is True
    assert prepared.config.read_bytes() == Path("configs/accuracy_campaign.json").read_bytes()
    mode = json.loads(prepared.run_mode.read_text())
    assert mode["code_manifest_sha256"]
    assert mode["execution_controls"] == {"calibration_grid_limit": None}
    assert "bootstrap_samples" not in mode["execution_controls"]
    assert mode["execution_identity"] == request.execution_identity
    assert mode["mode"] == "canonical"
    assert mode["schema_version"] == 3

    resumed = prepare_campaign_inputs(store, tmp_path / "resumed", request)
    assert resumed.run_mode.read_bytes() == prepared.run_mode.read_bytes()


def test_input_bootstrap_rejects_mode_or_control_drift(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    request = _request(tmp_path)
    prepare_campaign_inputs(store, tmp_path / "work", request)

    with pytest.raises(ValueError, match="input identity"):
        prepare_campaign_inputs(
            store,
            tmp_path / "drifted",
            _request(
                tmp_path,
                execution_identity={"kind": "local", "store": "/different/store"},
            ),
        )


def test_input_bootstrap_rejects_a_corrupt_published_identity(tmp_path: Path) -> None:
    root = tmp_path / "store"
    (root / "inputs").mkdir(parents=True)
    (root / "inputs/_COMPLETE.json").write_text("not-json")

    with pytest.raises(ValueError, match="established campaign input publication"):
        prepare_campaign_inputs(
            LocalArtifactStore(root),
            tmp_path / "work",
            _request(tmp_path),
        )


def test_input_bootstrap_can_recover_only_unpublished_stage_free_fragments(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    (root / "inputs").mkdir(parents=True)
    (root / "inputs/config.json").write_text("interrupted")
    store = LocalArtifactStore(root)

    prepared = prepare_campaign_inputs(store, tmp_path / "work", _request(tmp_path))

    assert prepared.config.read_bytes() == Path("configs/accuracy_campaign.json").read_bytes()
    assert store.verify_completion("inputs") is True


def test_input_evidence_probes_receive_the_absolute_deadline(tmp_path: Path) -> None:
    class RecordingStore(LocalArtifactStore):
        def __init__(self, root: Path) -> None:
            super().__init__(root, monotonic_clock=lambda: 10.0)
            self.explicit_deadlines: list[float | None] = []

        def exists(
            self,
            key: str,
            *,
            deadline_monotonic: float | None = None,
        ) -> bool:
            if deadline_monotonic is not None:
                self.explicit_deadlines.append(deadline_monotonic)
            return super().exists(key, deadline_monotonic=deadline_monotonic)

    store = RecordingStore(tmp_path / "store")

    prepare_campaign_inputs(
        store,
        tmp_path / "work",
        _request(tmp_path),
        deadline_monotonic=15.0,
    )

    assert store.explicit_deadlines == [15.0] * 13


def test_canonical_input_policy_rejects_non_null_calibration_grid_limit(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="canonical campaign controls"):
        prepare_campaign_inputs(
            LocalArtifactStore(tmp_path / "store"),
            tmp_path / "work",
            _request(
                tmp_path,
                calibration_grid_limit=1,
            ),
        )


def test_canonical_input_policy_requires_allowlisted_committed_config_bytes(
    tmp_path: Path,
) -> None:
    fake_canonical = tmp_path / "accuracy_campaign.json"
    fake_canonical.write_bytes(Path("configs/accuracy_campaign_cloud_reduced.json").read_bytes())

    with pytest.raises(ValueError, match="trusted committed canonical config"):
        prepare_campaign_inputs(
            LocalArtifactStore(tmp_path / "store"),
            tmp_path / "work",
            _request(
                tmp_path,
                canonical_config=fake_canonical,
                effective_config=fake_canonical,
            ),
        )


def test_verified_run_mode_binds_mode_claims_config_code_and_commit(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    request = _request(tmp_path)
    prepared = prepare_campaign_inputs(store, tmp_path / "work", request)

    verified = verify_campaign_run_mode(
        prepared.run_mode,
        canonical_config_path=request.canonical_config,
        config_path=prepared.config,
        code_manifest_path=prepared.code / "code.json",
        git_commit=request.git_commit,
    )

    assert verified["mode"] == "canonical"
    assert verified["scientific_claims_permitted"] is True


def test_verified_run_mode_rejects_float_schema_version(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    request = _request(tmp_path)
    prepared = prepare_campaign_inputs(store, tmp_path / "work", request)
    payload = json.loads(prepared.run_mode.read_text())
    payload["schema_version"] = 3.0
    prepared.run_mode.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="schema version is unsupported"):
        verify_campaign_run_mode(
            prepared.run_mode,
            canonical_config_path=request.canonical_config,
            config_path=prepared.config,
            code_manifest_path=prepared.code / "code.json",
            git_commit=request.git_commit,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mode", "reduced_non_scientific", "mode and claim policy"),
        ("scientific_claims_permitted", False, "mode and claim policy"),
        ("effective_config_sha256", "0" * 64, "configuration provenance"),
        ("code_manifest_sha256", "0" * 64, "code provenance"),
        ("git_commit", "b" * 40, "Git provenance"),
    ],
)
def test_verified_run_mode_rejects_provenance_or_claim_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    request = _request(tmp_path)
    prepared = prepare_campaign_inputs(store, tmp_path / "work", request)
    payload = json.loads(prepared.run_mode.read_text())
    payload[field] = value
    prepared.run_mode.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        verify_campaign_run_mode(
            prepared.run_mode,
            canonical_config_path=request.canonical_config,
            config_path=prepared.config,
            code_manifest_path=prepared.code / "code.json",
            git_commit=request.git_commit,
        )


def test_verified_run_mode_rejects_coordinated_reduced_to_canonical_rehash(
    tmp_path: Path,
) -> None:
    code = tmp_path / "code"
    code.mkdir()
    code_manifest = code / "code.json"
    code_manifest.write_text('{"name":"lp_3_7_16"}\n')
    effective = Path("configs/accuracy_campaign_cloud_reduced.json")
    path = tmp_path / "run-mode.json"
    payload = {
        "canonical_config": effective.name,
        "canonical_config_sha256": sha256_file(effective),
        "code_manifest_sha256": sha256_file(code_manifest),
        "effective_config_sha256": sha256_file(effective),
        "execution_controls": {"calibration_grid_limit": None},
        "execution_identity": {
            "kind": "local",
            "store": str((tmp_path / "store").resolve()),
        },
        "git_commit": "a" * 40,
        "mode": "canonical",
        "overrides": {},
        "schema_version": 3,
        "scientific_claims_permitted": True,
    }
    write_canonical_json(path, payload)

    with pytest.raises(ValueError, match="canonical config|canonical campaign controls"):
        verify_campaign_run_mode(
            path,
            canonical_config_path=Path("configs/accuracy_campaign.json"),
            config_path=effective,
            code_manifest_path=code_manifest,
            git_commit="a" * 40,
        )


def test_verified_run_mode_rejects_unvalidated_execution_identity(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    request = _request(tmp_path)
    prepared = prepare_campaign_inputs(store, tmp_path / "work", request)
    payload = json.loads(prepared.run_mode.read_text())
    payload["execution_identity"]["untrusted"] = True
    write_canonical_json(prepared.run_mode, payload)

    with pytest.raises(ValueError, match="execution identity"):
        verify_campaign_run_mode(
            prepared.run_mode,
            canonical_config_path=request.canonical_config,
            config_path=prepared.config,
            code_manifest_path=prepared.code / "code.json",
            git_commit=request.git_commit,
        )


def test_verified_run_mode_rejects_type_rehashed_reduced_override(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        calibration_grid_limit=1,
        campaign_mode="reduced_non_scientific",
        effective_config=Path("configs/accuracy_campaign_cloud_reduced.json"),
    )
    prepared = prepare_campaign_inputs(
        LocalArtifactStore(tmp_path / "store"),
        tmp_path / "work",
        request,
    )
    payload = json.loads(prepared.run_mode.read_text())
    assert payload["overrides"]["training_epochs"] == 1
    payload["overrides"]["training_epochs"] = True
    write_canonical_json(prepared.run_mode, payload)

    with pytest.raises(ValueError, match="policy does not match trusted inputs"):
        verify_campaign_run_mode(
            prepared.run_mode,
            canonical_config_path=request.canonical_config,
            config_path=prepared.config,
            code_manifest_path=prepared.code / "code.json",
            git_commit=request.git_commit,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("unexpected", True, "fields are incomplete"),
        ("outer_timeout_seconds", 28800.0, "deadline controls"),
        ("project", 7, "string fields"),
    ],
)
def test_cloud_execution_identity_requires_exact_fields_and_types(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    identity = _cloud_identity()
    identity[field] = value

    with pytest.raises(ValueError, match=message):
        prepare_campaign_inputs(
            LocalArtifactStore(tmp_path / "store"),
            tmp_path / "work",
            _request(tmp_path, execution_identity=identity),
        )
