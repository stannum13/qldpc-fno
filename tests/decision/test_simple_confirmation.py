from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from qldpc_fno.decision import planar_shot_data
from qldpc_fno.decision.simple_confirmation import (
    FIXTURE_SHOT_CONFIG,
    FROZEN_CONFIG,
    FROZEN_SHOT_CONFIG,
    SCIENTIFIC_DOMAIN,
    load_config,
    load_shot_config,
)

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "configs" / "simple_planar_confirmation.json"
_SHOTS = _ROOT / "configs" / "simple_planar_confirmation_shots.json"
_FIXTURE_SHOTS = _ROOT / "configs" / "simple_planar_confirmation_fixture_shots.json"


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload))
    return path


def test_config_freezes_every_confirmation_constant() -> None:
    config = load_config(_CONFIG)

    assert config == FROZEN_CONFIG
    assert config["contract_id"] == "simple_planar_confirmation_v1"
    assert config["policy_ids"] == ["fixed_rows_tol003", "margin_columns_chi8"]
    assert config["comparator_id"] == "fixed_columns_chi8"
    assert config["primary_endpoints"] == ["class_mismatch", "outcome_discordance"]
    assert config["primary_alpha"] == 0.00625
    assert config["discrepancy_budget"] == 0.005
    assert config["bootstrap_seed"] == 2713269656809941213
    assert config["bootstrap_replicates"] == 10_000
    assert config["minimum_work_saving"] == 0.5
    assert [item["action_id"] for item in config["actions"]] == [
        "rows_tol003",
        "columns_tol01",
        "rows_tol01",
        "columns_chi8",
        "exact_columns",
        "exact_rows",
    ]


def test_shot_configs_are_separate_frozen_identities() -> None:
    scientific = load_shot_config(_SHOTS, fixture=False)
    fixture = load_shot_config(_FIXTURE_SHOTS, fixture=True)

    assert scientific == FROZEN_SHOT_CONFIG
    assert fixture == FIXTURE_SHOT_CONFIG
    assert scientific["shots_per_rate"] == 2048
    assert fixture["shots_per_rate"] == 2
    assert scientific["seed_domain"] != fixture["seed_domain"]


@pytest.mark.parametrize("count", [2, 2047, 2049, True, 2048.0])
def test_scientific_count_is_frozen(tmp_path: Path, count: object) -> None:
    payload = dict(FROZEN_SHOT_CONFIG, shots_per_rate=count)
    path = _write_json(tmp_path / "shots.json", payload)
    with pytest.raises(ValueError):
        load_shot_config(path, fixture=False)


@pytest.mark.parametrize("count", [1, 3, 2048, True, 2.0])
def test_fixture_count_is_frozen(tmp_path: Path, count: object) -> None:
    payload = dict(FIXTURE_SHOT_CONFIG, shots_per_rate=count)
    path = _write_json(tmp_path / "shots.json", payload)
    with pytest.raises(ValueError):
        load_shot_config(path, fixture=True)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("contract_id", "changed"),
        ("required_shot_seed_domain", "changed"),
        ("fixture_seed_domain", "changed"),
        ("code_distance", 5.0),
        ("noise_model", "changed"),
        ("required_error_rates", [0.15, 0.1]),
        ("required_shots_per_rate", 2048.0),
        ("policy_ids", ["margin_columns_chi8", "fixed_rows_tol003"]),
        ("comparator_id", "fixed_rows_tol003"),
        ("margin_threshold", 0.3),
        ("reference_probability_tolerance", 1e-9),
        ("reference_log_ratio_tolerance", 1e-7),
        ("reference_margin_discrepancy_factor", 1.0),
        ("primary_endpoints", ["outcome_discordance", "class_mismatch"]),
        ("family_alpha", 0.1),
        ("primary_alpha", 0.0125),
        ("discrepancy_budget", 0.01),
        ("bootstrap_domain", "changed"),
        ("bootstrap_seed", 0),
        ("bootstrap_replicates", 9999),
        ("bootstrap_generator", "MT19937"),
        ("bootstrap_quantile", 0.1),
        ("bootstrap_quantile_method", "nearest"),
        ("minimum_work_saving", 0.49),
        ("order_rule", "changed"),
        ("invalidity_rule", "changed"),
        ("status_rule", "changed"),
    ],
)
def test_config_rejects_every_changed_constant(tmp_path: Path, key: str, value: object) -> None:
    payload = deepcopy(FROZEN_CONFIG)
    payload[key] = value
    with pytest.raises(ValueError):
        load_config(_write_json(tmp_path / "config.json", payload))


@pytest.mark.parametrize("key", list(FROZEN_CONFIG))
def test_config_rejects_every_missing_key(tmp_path: Path, key: str) -> None:
    payload = deepcopy(FROZEN_CONFIG)
    del payload[key]
    with pytest.raises(ValueError):
        load_config(_write_json(tmp_path / "config.json", payload))


def test_config_rejects_unknown_and_duplicate_keys(tmp_path: Path) -> None:
    payload = deepcopy(FROZEN_CONFIG)
    payload["unexpected"] = True
    with pytest.raises(ValueError):
        load_config(_write_json(tmp_path / "unknown.json", payload))

    duplicate = _CONFIG.read_text().replace(
        '"schema_version": 1,', '"schema_version": 1,\n  "schema_version": 1,', 1
    )
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        load_config(path)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_config_rejects_nonfinite_json_constants(tmp_path: Path, literal: str) -> None:
    text = _CONFIG.read_text().replace("0.30710401263493464", literal)
    path = tmp_path / "nonfinite.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize(
    "actions",
    [
        [],
        FROZEN_CONFIG["actions"][:-1],
        list(reversed(FROZEN_CONFIG["actions"])),
        [
            *FROZEN_CONFIG["actions"][:-1],
            {"action_id": "exact_rows", "mode": "rows", "chi": 8, "tol": None},
        ],
    ],
)
def test_config_rejects_changed_or_insufficient_action_inventory(
    tmp_path: Path, actions: object
) -> None:
    payload = deepcopy(FROZEN_CONFIG)
    payload["actions"] = actions
    with pytest.raises(ValueError):
        load_config(_write_json(tmp_path / "config.json", payload))


def test_action_rejects_unknown_missing_and_duplicate_fields(tmp_path: Path) -> None:
    for mutation in ("unknown", "missing"):
        payload = deepcopy(FROZEN_CONFIG)
        if mutation == "unknown":
            payload["actions"][0]["unexpected"] = True
        else:
            del payload["actions"][0]["mode"]
        with pytest.raises(ValueError):
            load_config(_write_json(tmp_path / f"{mutation}.json", payload))

    text = _CONFIG.read_text().replace(
        '"action_id": "rows_tol003",',
        '"action_id": "rows_tol003", "action_id": "rows_tol003",',
        1,
    )
    path = tmp_path / "duplicate-action.json"
    path.write_text(text)
    with pytest.raises(ValueError, match="duplicate"):
        load_config(path)


@pytest.mark.parametrize("fixture", [True, False])
def test_shot_config_rejects_wrong_mode_domain(tmp_path: Path, fixture: bool) -> None:
    payload = FROZEN_SHOT_CONFIG if fixture else FIXTURE_SHOT_CONFIG
    with pytest.raises(ValueError):
        load_shot_config(_write_json(tmp_path / "shots.json", payload), fixture=fixture)


@pytest.mark.parametrize("fixture", [None, 0, 1, "yes"])
def test_shot_config_requires_boolean_fixture_flag(tmp_path: Path, fixture: object) -> None:
    path = _write_json(tmp_path / "shots.json", FROZEN_SHOT_CONFIG)
    with pytest.raises(TypeError):
        load_shot_config(path, fixture=fixture)  # type: ignore[arg-type]


def test_shot_config_rejects_unknown_missing_duplicate_and_nonfinite(tmp_path: Path) -> None:
    payload = dict(FROZEN_SHOT_CONFIG, unexpected=True)
    with pytest.raises(ValueError):
        load_shot_config(_write_json(tmp_path / "unknown.json", payload), fixture=False)

    payload = dict(FROZEN_SHOT_CONFIG)
    del payload["noise_model"]
    with pytest.raises(ValueError):
        load_shot_config(_write_json(tmp_path / "missing.json", payload), fixture=False)

    duplicate = _SHOTS.read_text().replace(
        '"schema_version": 1,', '"schema_version": 1,\n  "schema_version": 1,', 1
    )
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        load_shot_config(path, fixture=False)

    path = tmp_path / "nonfinite.json"
    path.write_text(_SHOTS.read_text().replace("0.1", "NaN", 1))
    with pytest.raises(ValueError):
        load_shot_config(path, fixture=False)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema_version", True),
        ("campaign_seed", True),
        ("campaign_seed", 0),
        ("code_distance", 5.0),
        ("error_rates", [0.15, 0.1]),
        ("noise_model", "changed"),
    ],
)
def test_shot_config_rejects_changed_identity(
    tmp_path: Path, key: str, value: object
) -> None:
    payload = dict(FROZEN_SHOT_CONFIG)
    payload[key] = value
    with pytest.raises(ValueError):
        load_shot_config(_write_json(tmp_path / "shots.json", payload), fixture=False)


def test_fixture_cannot_derive_production_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_hash(*args, **kwargs):
        pytest.fail("must reject scientific test coordinates before hashing")

    monkeypatch.setattr(planar_shot_data.hashlib, "sha256", forbidden_hash)
    with pytest.raises(ValueError, match="scientific confirmation seed forbidden in test mode"):
        planar_shot_data._shot_seed(SCIENTIFIC_DOMAIN, error_rate=0.1, shot_index=0)


def test_fixture_coordinate_can_reach_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    real_hash = planar_shot_data.hashlib.sha256

    def recording_hash(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_hash(*args, **kwargs)

    monkeypatch.setattr(planar_shot_data.hashlib, "sha256", recording_hash)
    seed = planar_shot_data._shot_seed(
        FIXTURE_SHOT_CONFIG["seed_domain"], error_rate=0.1, shot_index=0
    )
    assert type(seed) is int
    assert calls == 1


def test_direct_generator_rejects_scientific_domain_before_sampler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from qecsim.models.generic import DepolarizingErrorModel

    def forbidden(*args, **kwargs):
        pytest.fail("scientific confirmation generator reached RNG or sampler")

    monkeypatch.setattr(planar_shot_data, "_shot_seed", forbidden)
    monkeypatch.setattr(DepolarizingErrorModel, "generate", forbidden)
    with pytest.raises(ValueError, match="scientific confirmation seed forbidden in test mode"):
        planar_shot_data.generate_planar_shots(_SHOTS, tmp_path / "out")
