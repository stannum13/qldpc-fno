from __future__ import annotations

import json
from pathlib import Path

import pytest

from qldpc_fno.campaign.inputs import CampaignInputRequest, prepare_campaign_inputs
from qldpc_fno.campaign.storage import LocalArtifactStore


def _request(tmp_path: Path, **overrides: object) -> CampaignInputRequest:
    code = tmp_path / "code"
    code.mkdir(exist_ok=True)
    (code / "code.json").write_text('{"name":"lp_3_7_16"}\n')
    values: dict[str, object] = {
        "bootstrap_samples": 10_000,
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


def test_shared_input_bootstrap_publishes_materializes_and_revalidates(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    request = _request(tmp_path)

    prepared = prepare_campaign_inputs(store, tmp_path / "work", request)

    assert store.verify_completion("inputs") is True
    assert prepared.config.read_bytes() == Path("configs/accuracy_campaign.json").read_bytes()
    mode = json.loads(prepared.run_mode.read_text())
    assert mode["code_manifest_sha256"]
    assert mode["execution_controls"] == {
        "bootstrap_samples": 10_000,
        "calibration_grid_limit": None,
    }
    assert mode["execution_identity"] == request.execution_identity
    assert mode["mode"] == "canonical"

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


@pytest.mark.parametrize(
    ("grid_limit", "bootstrap_samples"),
    [(1, 10_000), (None, 100), (1, 100)],
)
def test_canonical_input_policy_rejects_reduced_controls(
    tmp_path: Path,
    grid_limit: int | None,
    bootstrap_samples: int,
) -> None:
    with pytest.raises(ValueError, match="canonical campaign controls"):
        prepare_campaign_inputs(
            LocalArtifactStore(tmp_path / "store"),
            tmp_path / "work",
            _request(
                tmp_path,
                bootstrap_samples=bootstrap_samples,
                calibration_grid_limit=grid_limit,
            ),
        )
