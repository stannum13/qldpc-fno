from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from qldpc_fno.decision import planar_shot_data
from qldpc_fno.decision import simple_confirmation as confirmation
from qldpc_fno.decision.simple_confirmation import (
    FIXTURE_SHOT_CONFIG,
    FROZEN_CONFIG,
    FROZEN_SHOT_CONFIG,
    SCIENTIFIC_DOMAIN,
    bootstrap_saving,
    event_indicators,
    load_config,
    load_shot_config,
    primary_upper,
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


def test_mutating_exported_analysis_mapping_cannot_redefine_validator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = json.loads(_CONFIG.read_text())
    payload["margin_threshold"] = 0.25
    monkeypatch.setitem(confirmation.FROZEN_CONFIG, "margin_threshold", 0.25)

    with pytest.raises(ValueError):
        load_config(_write_json(tmp_path / "config.json", payload))


def test_mutating_exported_nested_lists_cannot_redefine_validator(
    tmp_path: Path,
) -> None:
    payload = json.loads(_CONFIG.read_text())
    payload["required_error_rates"][0] = 0.05
    exported_rates = confirmation.FROZEN_CONFIG["required_error_rates"]
    exported_rates[0] = 0.05
    try:
        with pytest.raises(ValueError):
            load_config(_write_json(tmp_path / "config.json", payload))
    finally:
        exported_rates[0] = 0.1


def test_mutating_exported_nested_action_cannot_redefine_validator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = json.loads(_CONFIG.read_text())
    payload["actions"][0]["tol"] = 0.004
    monkeypatch.setitem(confirmation.FROZEN_CONFIG["actions"][0], "tol", 0.004)

    with pytest.raises(ValueError):
        load_config(_write_json(tmp_path / "config.json", payload))


def test_mutating_exported_shot_mapping_cannot_redefine_validator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = json.loads(_SHOTS.read_text())
    payload["shots_per_rate"] = 1024
    monkeypatch.setitem(confirmation.FROZEN_SHOT_CONFIG, "shots_per_rate", 1024)

    with pytest.raises(ValueError):
        load_shot_config(_write_json(tmp_path / "shots.json", payload), fixture=False)


def test_mutating_exported_shot_nested_list_cannot_redefine_validator(
    tmp_path: Path,
) -> None:
    payload = json.loads(_FIXTURE_SHOTS.read_text())
    payload["error_rates"][1] = 0.2
    exported_rates = confirmation.FIXTURE_SHOT_CONFIG["error_rates"]
    exported_rates[1] = 0.2
    try:
        with pytest.raises(ValueError):
            load_shot_config(_write_json(tmp_path / "shots.json", payload), fixture=True)
    finally:
        exported_rates[1] = 0.15


def test_loaded_config_mutation_does_not_affect_later_load() -> None:
    first = load_config(_CONFIG)
    first["actions"][0]["tol"] = 0.004
    first["required_error_rates"][0] = 0.05

    second = load_config(_CONFIG)
    assert second["actions"][0]["tol"] == 0.003
    assert second["required_error_rates"] == [0.1, 0.15]


@pytest.mark.parametrize(
    ("export_name", "config_key", "changed"),
    [
        ("SCIENTIFIC_DOMAIN", "required_shot_seed_domain", "changed/scientific"),
        ("FIXTURE_DOMAIN", "fixture_seed_domain", "changed/fixture"),
        ("BOOTSTRAP_DOMAIN", "bootstrap_domain", "changed/bootstrap"),
        ("BOOTSTRAP_SEED", "bootstrap_seed", 1),
        ("MARGIN_THRESHOLD", "margin_threshold", 0.25),
    ],
)
def test_rebinding_exported_analysis_scalar_cannot_redefine_validator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    export_name: str,
    config_key: str,
    changed: object,
) -> None:
    payload = json.loads(_CONFIG.read_text())
    payload[config_key] = changed
    monkeypatch.setattr(confirmation, export_name, changed)

    with pytest.raises(ValueError):
        load_config(_write_json(tmp_path / "config.json", payload))


@pytest.mark.parametrize(
    ("fixture", "export_name", "config_key", "changed"),
    [
        (False, "SCIENTIFIC_DOMAIN", "seed_domain", "changed/scientific"),
        (False, "SCIENTIFIC_CAMPAIGN_SEED", "campaign_seed", 1),
        (True, "FIXTURE_DOMAIN", "seed_domain", "changed/fixture"),
        (True, "FIXTURE_CAMPAIGN_SEED", "campaign_seed", 1),
    ],
)
def test_rebinding_exported_shot_scalar_cannot_redefine_validator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fixture: bool,
    export_name: str,
    config_key: str,
    changed: object,
) -> None:
    source = _FIXTURE_SHOTS if fixture else _SHOTS
    payload = json.loads(source.read_text())
    payload[config_key] = changed
    monkeypatch.setattr(confirmation, export_name, changed)

    with pytest.raises(ValueError):
        load_shot_config(_write_json(tmp_path / "shots.json", payload), fixture=fixture)


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


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        (0, 0.0024750442291897544),
        (1, 0.003498836715834498),
        (2, 0.004385297715096555),
        (3, 0.005205093018026918),
        (2048, 1.0),
    ],
)
def test_primary_upper_matches_independent_binomial_inversion(
    events: int, expected: float
) -> None:
    actual = primary_upper(events, 2048)
    assert actual == pytest.approx(expected, abs=1e-15)
    if events < 2048:
        cdf = sum(
            math.comb(2048, j) * actual**j * (1.0 - actual) ** (2048 - j)
            for j in range(events + 1)
        )
        assert cdf == pytest.approx(0.00625, abs=1e-13)


def test_primary_upper_is_monotone_and_freezes_confirmation_cutoff() -> None:
    bounds = [primary_upper(events, 2048) for events in range(5)]
    assert bounds == sorted(bounds)
    assert [bound <= 0.005 for bound in bounds[:4]] == [True, True, True, False]


@pytest.mark.parametrize(
    ("events", "shots"),
    [
        (True, 2048),
        (0, True),
        (0.0, 2048),
        (0, 2048.0),
        (-1, 2048),
        (2049, 2048),
        (0, 0),
        (0, -1),
    ],
)
def test_primary_upper_rejects_noninteger_or_out_of_range_counts(
    events: object, shots: object
) -> None:
    with pytest.raises(ValueError):
        primary_upper(events, shots)  # type: ignore[arg-type]


def test_two_wrong_classes_can_share_failure_outcome() -> None:
    assert event_indicators(
        reference_valid=True,
        policy_valid=True,
        policy_class=1,
        reference_class=2,
        policy_failure=True,
        reference_failure=True,
    ) == (True, False)


@pytest.mark.parametrize(
    ("policy_class", "reference_class", "policy_failure", "reference_failure", "expected"),
    [
        (0, 0, False, False, (False, False)),
        (1, 2, False, False, (True, False)),
        (3, 3, True, False, (False, True)),
        (0, 3, False, True, (True, True)),
    ],
)
def test_event_indicators_keep_class_and_failure_endpoints_distinct(
    policy_class: int,
    reference_class: int,
    policy_failure: bool,
    reference_failure: bool,
    expected: tuple[bool, bool],
) -> None:
    assert event_indicators(
        reference_valid=True,
        policy_valid=True,
        policy_class=policy_class,
        reference_class=reference_class,
        policy_failure=policy_failure,
        reference_failure=reference_failure,
    ) == expected


@pytest.mark.parametrize(
    ("reference_valid", "policy_valid"),
    [(False, True), (True, False), (False, False)],
)
def test_invalid_reference_or_policy_counts_both_primary_events(
    reference_valid: bool, policy_valid: bool
) -> None:
    assert event_indicators(
        reference_valid=reference_valid,
        policy_valid=policy_valid,
        policy_class=0 if policy_valid else None,
        reference_class=0 if reference_valid else None,
        policy_failure=False if policy_valid else None,
        reference_failure=False if reference_valid else None,
    ) == (True, True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_valid": 1},
        {"policy_valid": 0},
        {"policy_class": True},
        {"policy_class": -1},
        {"policy_class": 4},
        {"reference_class": 1.0},
        {"policy_failure": 0},
        {"reference_failure": 1},
        {"policy_class": None},
        {"reference_failure": None},
    ],
)
def test_event_indicators_reject_invalid_validity_and_labels(kwargs: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "reference_valid": True,
        "policy_valid": True,
        "policy_class": 0,
        "reference_class": 0,
        "policy_failure": False,
        "reference_failure": False,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        event_indicators(**arguments)  # type: ignore[arg-type]


def test_event_indicators_require_null_labels_for_invalid_records() -> None:
    with pytest.raises(ValueError):
        event_indicators(
            reference_valid=False,
            policy_valid=True,
            policy_class=0,
            reference_class=0,
            policy_failure=False,
            reference_failure=False,
        )


def _paired(*, margin: float, column: float = 100.0, rows: float = 40.0) -> np.ndarray:
    paired = np.zeros((2, 2048, 11), dtype=np.float64)
    paired[:, :, 0] = margin
    paired[:, :, 1] = column
    paired[:, :, 2] = rows
    return paired


def test_bootstrap_constant_seventy_five_percent_saving_is_exact() -> None:
    result = bootstrap_saving(_paired(margin=25.0))

    assert set(result) == {
        "status",
        "estimand",
        "replicates",
        "seed",
        "bit_generator",
        "quantile_method",
        "estimate",
        "lower_bound",
        "threshold",
        "passed",
        "unavailable_reason",
    }
    assert result == {
        "status": "available",
        "estimand": "equal_rate_mean_paired_relative_saving",
        "replicates": 10_000,
        "seed": 2713269656809941213,
        "bit_generator": "PCG64",
        "quantile_method": "linear",
        "estimate": 0.75,
        "lower_bound": 0.75,
        "threshold": 0.5,
        "passed": True,
        "unavailable_reason": None,
    }


@pytest.mark.parametrize(
    ("margin", "estimate", "passed"),
    [(50.0, 0.5, False), (125.0, -0.25, False)],
)
def test_bootstrap_gate_is_strict_and_preserves_negative_savings(
    margin: float, estimate: float, passed: bool
) -> None:
    result = bootstrap_saving(_paired(margin=margin))
    assert result["estimate"] == pytest.approx(estimate)
    assert result["lower_bound"] == pytest.approx(estimate)
    assert result["passed"] is passed


def test_bootstrap_equal_weights_rates_and_uses_mean_of_paired_ratios() -> None:
    paired = _paired(margin=25.0)
    paired[0, :, 0] = 0.0
    paired[1, :, 0] = 100.0
    paired[0, :, 1] = np.tile([4.0, 100.0], 1024)
    paired[1, :, 1] = np.tile([4.0, 100.0], 1024)
    paired[0, :, 0] = paired[0, :, 1] * 0.25
    paired[1, :, 0] = paired[1, :, 1] * 0.75

    result = bootstrap_saving(paired)
    ratio_of_totals = 1.0 - float(np.sum(paired[:, :, 0])) / float(np.sum(paired[:, :, 1]))
    assert result["estimate"] == pytest.approx(0.5)
    assert ratio_of_totals == pytest.approx(0.5)
    assert result["passed"] is False


def test_bootstrap_mean_ratios_differs_from_ratio_of_totals_when_costs_covary() -> None:
    paired = _paired(margin=25.0)
    for stratum in range(2):
        paired[stratum, :1024, 0] = 0.0
        paired[stratum, :1024, 1] = 1.0
        paired[stratum, 1024:, 0] = 100.0
        paired[stratum, 1024:, 1] = 100.0

    result = bootstrap_saving(paired)
    ratio_of_totals = 1.0 - float(np.sum(paired[:, :, 0])) / float(np.sum(paired[:, :, 1]))
    assert result["estimate"] == pytest.approx(0.5)
    assert ratio_of_totals == pytest.approx(1.0 / 101.0)


def test_bootstrap_is_deterministic_and_pairing_sensitive() -> None:
    paired = _paired(margin=25.0)
    ramp = np.arange(1, 2049, dtype=np.float64)
    paired[:, :, 1] = ramp * 10.0
    paired[:, :, 0] = ramp * np.tile([1.0, 9.0], 1024)
    first = bootstrap_saving(paired)
    second = bootstrap_saving(paired.copy())
    unpaired = paired.copy()
    unpaired[:, :, 0] = unpaired[:, ::-1, 0]
    changed = bootstrap_saving(unpaired)

    assert first == second
    assert changed["estimate"] != pytest.approx(first["estimate"])


def test_bootstrap_draws_complete_paired_rows_with_frozen_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int, int]] = []
    paired_row_slices: list[tuple[object, object, object]] = []
    original_asarray = confirmation.np.asarray

    class TrackingArray(np.ndarray):
        def __getitem__(self, key):
            if (
                isinstance(key, tuple)
                and len(key) == 3
                and type(key[0]) is int
                and isinstance(key[1], np.ndarray)
            ):
                paired_row_slices.append(key)
            return super().__getitem__(key)

    class GeneratorSpy:
        def __init__(self, bit_generator: object) -> None:
            assert type(bit_generator).__name__ == "PCG64"

        def integers(self, low: int, high: int, *, size: int) -> np.ndarray:
            calls.append((low, high, size))
            return np.arange(size, dtype=np.int64)

    def tracking_asarray(value: object, *, dtype: object) -> TrackingArray:
        return original_asarray(value, dtype=dtype).view(TrackingArray)

    monkeypatch.setattr(confirmation.np.random, "Generator", GeneratorSpy)
    monkeypatch.setattr(confirmation.np, "asarray", tracking_asarray)
    result = bootstrap_saving(_paired(margin=25.0))

    assert result["estimate"] == 0.75
    assert result["lower_bound"] == 0.75
    assert calls == [(0, 2048, 2048)] * 20_000
    assert len(paired_row_slices) == 20_000
    assert all(key[2] == slice(None) for key in paired_row_slices)


@pytest.mark.parametrize(
    ("mutator", "reason"),
    [
        (lambda value: value.__setitem__((0, 0, 0), np.nan), "nonfinite_values"),
        (lambda value: value.__setitem__((0, 0, 0), -1.0), "negative_work"),
        (lambda value: value.__setitem__((0, 0, 0), 1.5), "noninteger_work"),
        (lambda value: value.__setitem__((0, 0, 1), 0.0), "nonpositive_comparator_work"),
    ],
)
def test_bootstrap_invalid_or_missing_costs_suppress_inference(mutator, reason: str) -> None:
    paired = _paired(margin=25.0)
    mutator(paired)
    result = bootstrap_saving(paired)
    assert result["status"] == "unavailable"
    assert result["estimate"] is None
    assert result["lower_bound"] is None
    assert result["passed"] is False
    assert result["unavailable_reason"] == reason


@pytest.mark.parametrize(
    "paired",
    [
        np.zeros((2048, 11)),
        np.zeros((2, 2047, 11)),
        np.zeros((2, 2048, 10)),
        np.zeros((2, 2048, 11), dtype=object),
    ],
)
def test_bootstrap_rejects_wrong_shape_or_nonnumeric_array(paired: np.ndarray) -> None:
    with pytest.raises(ValueError):
        bootstrap_saving(paired)


@pytest.mark.parametrize("shots", [2, 2048])
def test_bootstrap_rejects_complex_arrays_before_imaginary_work_is_discarded(
    shots: int,
) -> None:
    paired = np.zeros((2, shots, 11), dtype=np.complex128)
    paired[:, :, 0] = 25.0 + 999.0j
    paired[:, :, 1] = 100.0

    with pytest.raises(ValueError, match="real numeric"):
        bootstrap_saving(paired)


def test_bootstrap_rejects_non_array_input() -> None:
    with pytest.raises(TypeError):
        bootstrap_saving([])  # type: ignore[arg-type]


def test_bootstrap_fixture_shape_never_runs_inference() -> None:
    paired = np.zeros((2, 2, 11), dtype=np.float64)
    paired[:, :, 0] = 25.0
    paired[:, :, 1] = 100.0
    paired[:, :, 2] = 40.0
    result = bootstrap_saving(paired)
    assert result["status"] == "fixture_only"
    assert result["estimate"] is None
    assert result["lower_bound"] is None
    assert result["passed"] is False
    assert result["unavailable_reason"] == "fixture_analysis_is_noninferential"


@pytest.mark.parametrize(
    ("index", "value", "reason"),
    [
        ((0, 0, 0), np.nan, "nonfinite_values"),
        ((0, 0, 0), -1.0, "negative_work"),
        ((0, 0, 0), 1.5, "noninteger_work"),
        ((0, 0, 1), 0.0, "nonpositive_comparator_work"),
        ((0, 0, 3), 2.0, "nonbinary_outcomes"),
    ],
)
def test_bootstrap_fixture_contents_are_validated_before_fixture_status(
    index: tuple[int, int, int], value: float, reason: str
) -> None:
    paired = np.zeros((2, 2, 11), dtype=np.float64)
    paired[:, :, 0] = 25.0
    paired[:, :, 1] = 100.0
    paired[:, :, 2] = 40.0
    paired[index] = value

    result = bootstrap_saving(paired)
    assert result["status"] == "unavailable"
    assert result["unavailable_reason"] == reason
    assert result["estimate"] is None
    assert result["lower_bound"] is None
    assert result["passed"] is False
