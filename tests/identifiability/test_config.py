from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from qldpc_fno.identifiability.config import load_identifiability_config

CONFIG_PATH = Path("configs/temporal_identifiability.json")


@pytest.fixture
def canonical_payload() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text())


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return path


def test_canonical_config_freezes_the_preregistered_scientific_contract() -> None:
    config = load_identifiability_config(CONFIG_PATH)

    assert config.schema_version == 1
    assert config.artifact_label == "confirmatory_gate"
    assert config.regimes == ("stationary_iid", "temporal_uniform")
    assert config.roles == ("train", "validation", "calibration", "test")
    assert (
        config.splits.train,
        config.splits.validation,
        config.splits.calibration,
        config.splits.test,
    ) == (8, 8, 8, 64)
    assert (config.rounds.burn_in, config.rounds.scored) == (64, 128)
    assert (
        config.dynamics.ar_coefficient,
        config.dynamics.innovation_std,
        config.dynamics.clip,
    ) == (0.97, 0.08, 1.20)
    assert config.dynamics.base_probability == 0.0375
    assert config.dynamics.probability_clip == (1e-5, 0.25)

    assert config.grid.interior_cells == 2048
    assert config.grid.doubled_interior_cells == 4096
    assert config.grid.open_loop_interior_cells == 4096
    assert config.grid.convergence_tolerance == 2.5e-5
    assert config.inference.delta_nll == 0.00025
    assert config.inference.bootstrap_draws == 10_000
    assert config.inference.one_sided_alpha == 0.05
    assert config.inference.holm_alpha == 0.05
    assert config.inference.nmse_denominator == pytest.approx(0.08**2 / (1 - 0.97**2))
    assert config.inference.calibration_bins == 10
    assert config.inference.calibration_range == (1e-5, 0.25)

    assert config.fisher.draws == 10_000
    assert config.fisher.draw_law == "stationary_normal_then_clip"
    assert config.fisher.finite_difference_step == 1e-6
    assert config.fisher.absolute_tolerance == 1e-8
    assert config.fisher.relative_tolerance == 1e-6
    assert config.runtime.process_cpu_seconds == 21_600

    assert config.baselines.empirical_stationary_shrinkage == (1.0,)
    assert config.baselines.ewma_decays == (0.5, 0.8, 0.9, 0.97, 0.99)
    assert config.baselines.ewma_kernel == 5
    assert config.baselines.logistic_lags == 32
    assert config.baselines.logistic_kernel == 3
    assert config.baselines.logistic_l2 == (1e-4, 1e-3, 1e-2, 1e-1)
    assert config.baselines.lbfgs_max_iter == 500
    assert config.baselines.tie_rule == "sorted_grid_first_minimum"
    assert config.baselines.tie_tolerance == 1e-12
    assert config.baselines.calibration == "identity"
    assert config.baselines.arm_aliases == (
        ("known_marginal", "known_marginal"),
        ("empirical_stationary", "empirical_stationary"),
        ("ewma", "ewma"),
        ("logistic_ar32", "logistic_ar32"),
        ("parity_moment_ar", "parity_moment_ar"),
        ("grid_bayes", "grid_bayes"),
        ("latent_history_oracle", "latent_history_oracle"),
        ("contemporaneous_oracle", "contemporaneous_oracle"),
    )

    assert (
        config.decoder.max_iter,
        config.decoder.bp_method,
        config.decoder.schedule,
        config.decoder.ms_scaling_factor,
        config.decoder.lsd_method,
        config.decoder.lsd_order,
    ) == (100, "minimum_sum", "serial", 0.0, "LSD_E", 5)

    assert config.code.name == "lp_3_7_16"
    assert (config.code.ell, config.code.n, config.code.k) == (45, 2610, 744)
    assert config.code.hx_sha256 == (
        "fc685627e7a7139b6af9c12187879a02f19c46598fe99a9be816f58d4627ead8"
    )
    assert config.code.hz_sha256 == (
        "82b536419e91a3c877685d6ab347810878a46dfbff76373d5a02b160bb8d2ecb"
    )

    with pytest.raises(FrozenInstanceError):
        config.rounds.scored = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "section",
    [
        "code",
        "splits",
        "rounds",
        "dynamics",
        "grid",
        "fisher",
        "inference",
        "baselines",
        "decoder",
        "runtime",
        "seeds",
    ],
)
def test_loader_rejects_missing_nested_keys(
    tmp_path: Path, canonical_payload: dict[str, object], section: str
) -> None:
    nested = canonical_payload[section]
    assert isinstance(nested, dict)
    nested.pop(next(iter(nested)))

    with pytest.raises(ValueError, match="missing"):
        load_identifiability_config(_write(tmp_path, canonical_payload))


def test_loader_rejects_unknown_top_level_and_nested_keys(
    tmp_path: Path, canonical_payload: dict[str, object]
) -> None:
    canonical_payload["development_mode"] = True
    with pytest.raises(ValueError, match="unknown"):
        load_identifiability_config(_write(tmp_path, canonical_payload))

    canonical_payload.pop("development_mode")
    dynamics = canonical_payload["dynamics"]
    assert isinstance(dynamics, dict)
    dynamics["spatial_amplitude"] = 0.1
    with pytest.raises(ValueError, match="unknown"):
        load_identifiability_config(_write(tmp_path, canonical_payload))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("splits", "train"), True),
        (("rounds", "burn_in"), False),
        (("grid", "interior_cells"), True),
        (("runtime", "process_cpu_seconds"), True),
        (("dynamics", "innovation_std"), float("nan")),
        (("grid", "convergence_tolerance"), float("inf")),
        (("fisher", "absolute_tolerance"), float("-inf")),
    ],
)
def test_loader_rejects_boolean_integers_and_nonfinite_numbers(
    tmp_path: Path,
    canonical_payload: dict[str, object],
    path: tuple[str, str],
    value: object,
) -> None:
    section = canonical_payload[path[0]]
    assert isinstance(section, dict)
    section[path[1]] = value
    with pytest.raises(ValueError):
        load_identifiability_config(_write(tmp_path, canonical_payload))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("splits", "test"), 63),
        (("rounds", "scored"), 127),
        (("dynamics", "ar_coefficient"), 0.96),
        (("grid", "open_loop_interior_cells"), 2048),
        (("fisher", "draw_law"), "clipped_stationary_normal"),
        (("inference", "calibration_bins"), 9),
        (("baselines", "empirical_stationary_shrinkage"), [0.0, 1.0]),
        (("decoder", "lsd_order"), 4),
        (("runtime", "process_cpu_seconds"), 21_599),
        (("seeds", "campaign"), 0),
    ],
)
def test_public_loader_rejects_altered_scientific_constants(
    tmp_path: Path,
    canonical_payload: dict[str, object],
    path: tuple[str, str],
    value: object,
) -> None:
    section = canonical_payload[path[0]]
    assert isinstance(section, dict)
    section[path[1]] = value
    with pytest.raises(ValueError, match="canonical|must equal|fixed"):
        load_identifiability_config(_write(tmp_path, canonical_payload))


def test_loader_rejects_mutable_or_reordered_regimes_and_bad_artifact_label(
    tmp_path: Path, canonical_payload: dict[str, object]
) -> None:
    canonical_payload["regimes"] = ["temporal_uniform", "stationary_iid"]
    with pytest.raises(ValueError, match="regimes"):
        load_identifiability_config(_write(tmp_path, canonical_payload))

    canonical_payload["regimes"] = ["stationary_iid", "temporal_uniform"]
    canonical_payload["artifact_label"] = "development"
    with pytest.raises(ValueError, match="artifact_label"):
        load_identifiability_config(_write(tmp_path, canonical_payload))


def test_loader_rejects_noncanonical_code_metadata(
    tmp_path: Path, canonical_payload: dict[str, object]
) -> None:
    code = canonical_payload["code"]
    assert isinstance(code, dict)
    code["name"] = "lp_3_7_15"
    with pytest.raises(ValueError, match="canonical"):
        load_identifiability_config(_write(tmp_path, canonical_payload))
