import json
from pathlib import Path

import pytest

from qldpc_fno.temporal.config import CausalExperimentConfig

CONFIG_PATH = Path("configs/causal_fno_hippo_reduced.json")


def _payload() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text())


def _write_config(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return path


def test_reduced_config_matches_the_causal_experiment_contract() -> None:
    config = CausalExperimentConfig.from_json(CONFIG_PATH)

    assert config.artifact_mode == "reduced_non_scientific"
    assert config.regimes == (
        "stationary_iid",
        "static_spatial_latent",
        "temporal_uniform",
        "joint_in_basis",
        "joint_basis_mismatch",
    )
    assert config.code.name == "lp_3_7_16"
    assert (config.code.ell, config.code.n, config.code.k) == (45, 2610, 744)
    assert (config.splits.train, config.splits.validation) == (2, 2)
    assert (config.splits.calibration, config.splits.test) == (2, 0)
    assert (config.rounds.burn_in, config.rounds.scored) == (8, 16)
    assert config.generator.base_probability == 0.0375
    assert config.generator.probability_bounds == (1e-5, 0.25)
    assert config.generator.spatial_frequency_range == (1, 3)
    assert config.generator.joint_base_amplitude == 0.45
    assert config.generator.burst_profile_concentration == 4.0
    assert config.generator.mismatch_step_range == (-1, 1)
    assert config.model.hidden_width == 32
    assert config.model.fno_modes == 12
    assert config.model.fir_history == 32
    assert config.model.hippo_order == 16
    assert config.decoder.schedule == "serial"
    assert config.decoder.lsd_method == "LSD_E"
    assert config.decoder.lsd_order == 5
    assert config.optimizer.training_seed == 1701
    assert config.overfit_fixture.seed == 1801
    assert (config.overfit_fixture.sequences, config.overfit_fixture.burn_in) == (2, 8)
    assert config.overfit_fixture.scored == 16
    assert config.overfit_fixture.learning_rate == 0.01
    assert config.overfit_fixture.max_steps == 2000
    assert config.overfit_fixture.nll_threshold == 0.03
    assert config.overfit_fixture.accuracy_threshold == 0.995


def test_config_round_trips_without_losing_schema_information() -> None:
    config = CausalExperimentConfig.from_json(CONFIG_PATH)
    assert CausalExperimentConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ell", 45.0),
        ("ell", True),
        ("n", 2610.0),
        ("n", True),
        ("k", 744.0),
        ("k", True),
        ("distance_upper_bound", 16.0),
        ("distance_upper_bound", True),
    ],
)
def test_config_rejects_non_integer_code_identity_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = _payload()
    payload["code"][field] = value

    with pytest.raises(ValueError, match=rf"{field} must be an integer"):
        CausalExperimentConfig.from_json(_write_config(tmp_path, payload))


@pytest.mark.parametrize("name", [123, "lp_3_7_15"])
def test_config_requires_exact_string_code_name(tmp_path: Path, name: object) -> None:
    payload = _payload()
    payload["code"]["name"] = name

    with pytest.raises(ValueError, match="name must be 'lp_3_7_16'"):
        CausalExperimentConfig.from_json(_write_config(tmp_path, payload))


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda payload: payload.update({"surprise": 1}), "unknown fields"),
        (lambda payload: payload.pop("rounds"), "missing fields"),
        (lambda payload: payload["model"].update({"surprise": 1}), "unknown fields"),
        (lambda payload: payload["generator"].pop("base_probability"), "missing fields"),
    ],
)
def test_config_rejects_unknown_and_missing_fields(
    tmp_path: Path, mutate, match: str
) -> None:
    payload = _payload()
    mutate(payload)

    with pytest.raises(ValueError, match=match):
        CausalExperimentConfig.from_json(_write_config(tmp_path, payload))


@pytest.mark.parametrize(
    ("section", "field", "value", "match"),
    [
        ("generator", "base_probability", 0.0, "base_probability"),
        ("generator", "min_probability", 0.3, "probability bounds"),
        ("generator", "global_ar_coefficient", 1.0, "global_ar_coefficient"),
        ("generator", "burst_start_probability", 1.1, "burst_start_probability"),
        ("rounds", "burn_in", -1, "burn_in"),
        ("rounds", "scored", 0, "scored"),
        ("model", "fno_modes", 24, "fno_modes"),
        ("optimizer", "learning_rate", float("nan"), "learning_rate"),
        ("decoder", "lsd_order", -1, "lsd_order"),
    ],
)
def test_config_rejects_invalid_values(
    tmp_path: Path, section: str, field: str, value: object, match: str
) -> None:
    payload = _payload()
    payload[section][field] = value

    with pytest.raises(ValueError, match=match):
        CausalExperimentConfig.from_json(_write_config(tmp_path, payload))


def test_reduced_config_requires_non_scientific_artifact_label(tmp_path: Path) -> None:
    payload = _payload()
    payload["artifact_mode"] = "scientific"

    with pytest.raises(ValueError, match="reduced_non_scientific"):
        CausalExperimentConfig.from_json(_write_config(tmp_path, payload))


def test_config_rejects_missing_decoder_schedule(tmp_path: Path) -> None:
    payload = _payload()
    del payload["decoder"]["schedule"]

    with pytest.raises(ValueError, match="missing fields.*schedule"):
        CausalExperimentConfig.from_json(_write_config(tmp_path, payload))


def test_config_rejects_invalid_decoder_schedule(tmp_path: Path) -> None:
    payload = _payload()
    payload["decoder"]["schedule"] = "parallel"

    with pytest.raises(ValueError, match="schedule must be 'serial'"):
        CausalExperimentConfig.from_json(_write_config(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_iter", 101),
        ("ms_scaling_factor", 0.5),
        ("lsd_order", 4),
    ],
)
def test_config_requires_exact_canonical_bplsd_policy(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = _payload()
    payload["decoder"][field] = value

    with pytest.raises(ValueError, match="canonical BP-LSD"):
        CausalExperimentConfig.from_json(_write_config(tmp_path, payload))


@pytest.mark.parametrize("field,value", [("mismatch_step_min", -2), ("mismatch_step_max", 2)])
def test_config_rejects_mismatch_steps_outside_supported_range(
    tmp_path: Path, field: str, value: int
) -> None:
    payload = _payload()
    payload["generator"][field] = value

    with pytest.raises(ValueError, match=r"mismatch steps must be within \[-1, 1\]"):
        CausalExperimentConfig.from_json(_write_config(tmp_path, payload))


@pytest.mark.parametrize("field,value", [("mismatch_step_min", -1), ("mismatch_step_max", 1)])
def test_config_accepts_mismatch_step_boundaries(
    tmp_path: Path, field: str, value: int
) -> None:
    payload = _payload()
    payload["generator"][field] = value

    config = CausalExperimentConfig.from_json(_write_config(tmp_path, payload))
    assert getattr(config.generator, field) == value


@pytest.mark.parametrize("artifact_mode", [[], {}])
def test_config_rejects_non_string_artifact_mode(
    tmp_path: Path, artifact_mode: object
) -> None:
    payload = _payload()
    payload["artifact_mode"] = artifact_mode

    with pytest.raises(ValueError, match="artifact_mode"):
        CausalExperimentConfig.from_json(_write_config(tmp_path, payload))


def test_config_rejects_missing_or_duplicate_regime_names(tmp_path: Path) -> None:
    payload = _payload()
    payload["regimes"][-1] = "joint_in_basis"

    with pytest.raises(ValueError, match="five named regimes"):
        CausalExperimentConfig.from_json(_write_config(tmp_path, payload))
