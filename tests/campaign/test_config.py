import json
from pathlib import Path

import pytest

from qldpc_fno.campaign.config import CampaignConfig


def _canonical_payload() -> dict[str, object]:
    payload = json.loads(Path("configs/accuracy_campaign.json").read_text())
    payload["selection_mode"] = "pilot"
    payload["test_stopping_mode"] = "adaptive"
    return payload


def test_canonical_config_has_disjoint_roles_and_bounded_cloud_job() -> None:
    config = CampaignConfig.from_json(Path("configs/accuracy_campaign.json"))
    assert config.noise_grid == (0.003, 0.005, 0.008, 0.012, 0.018, 0.025)
    assert config.cloud_cpu == 8
    assert config.cloud_memory == "32Gi"
    assert config.cloud_timeout_seconds == 8 * 60 * 60
    assert config.checkpoint_grace_seconds == 45 * 60
    assert config.calibration_decode_shots_cap == 512
    assert config.calibration_shortlist_per_method == 4
    assert config.max_test_shots_per_point == 200_000
    assert config.target_failures == 200


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"unknown": "field"}, "unknown fields"),
        ({"noise_grid": [0.003, 0.003]}, "strictly increasing"),
        ({"noise_grid": [0.003, 0.5]}, "between 0 and 0.5"),
        ({"target_failures": 200_001}, "target_failures"),
        ({"calibration_decode_shots_cap": 10_001}, "calibration_decode_shots_cap"),
        ({"calibration_shortlist_per_method": 49}, "calibration_shortlist_per_method"),
        ({"checkpoint_grace_seconds": 2_699}, "at least 2700"),
        ({"checkpoint_grace_seconds": 28_800}, "checkpoint_grace_seconds"),
    ],
)
def test_config_rejects_invalid_policy(
    tmp_path: Path, changes: dict[str, object], match: str
) -> None:
    payload = json.loads(Path("configs/accuracy_campaign.json").read_text())
    payload.update(changes)
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=match):
        CampaignConfig.from_json(path)


def test_config_rejects_missing_field(tmp_path: Path) -> None:
    payload = json.loads(Path("configs/accuracy_campaign.json").read_text())
    del payload["campaign_seed"]
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="missing fields"):
        CampaignConfig.from_json(path)


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_config_rejects_non_finite_noise_grid_entries(tmp_path: Path, non_finite: float) -> None:
    payload = json.loads(Path("configs/accuracy_campaign.json").read_text())
    payload["noise_grid"][2] = non_finite
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="noise_grid values must be finite"):
        CampaignConfig.from_json(path)


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_config_rejects_non_finite_training_learning_rate(
    tmp_path: Path, non_finite: float
) -> None:
    payload = json.loads(Path("configs/accuracy_campaign.json").read_text())
    payload["training_learning_rate"] = non_finite
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="training_learning_rate must be finite"):
        CampaignConfig.from_json(path)


def test_config_accepts_fixed_selection_and_stopping(tmp_path: Path) -> None:
    payload = _canonical_payload()
    payload["noise_grid"] = [0.0375]
    payload["selection_mode"] = "fixed"
    payload["test_stopping_mode"] = "fixed"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    config = CampaignConfig.from_json(path)
    assert config.selection_mode == "fixed"
    assert config.test_stopping_mode == "fixed"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("selection_mode", "manual", "selection_mode"),
        ("test_stopping_mode", "deadline", "test_stopping_mode"),
    ],
)
def test_config_rejects_invalid_campaign_modes(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    payload = _canonical_payload()
    payload[field] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        CampaignConfig.from_json(path)


@pytest.mark.parametrize("field", ["selection_mode", "test_stopping_mode"])
@pytest.mark.parametrize("value", [None, True, 1, [], {}])
def test_config_rejects_non_string_campaign_modes(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = _canonical_payload()
    payload[field] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=rf"{field} must be"):
        CampaignConfig.from_json(path)


def test_disconfirming_config_is_strict_and_fixed() -> None:
    config = CampaignConfig.from_json(Path("configs/accuracy_disconfirm_p0375.json"))
    assert config.noise_grid == (0.0375,)
    assert config.selection_mode == "fixed"
    assert config.test_stopping_mode == "fixed"
    assert config.max_test_shots_per_point == 2048
