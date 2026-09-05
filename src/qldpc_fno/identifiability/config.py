"""Strict preregistered configuration for temporal identifiability."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from qldpc_fno.decoders.bplsd import BPLSDConfig

REGIMES = ("stationary_iid", "temporal_uniform")
ROLES = ("train", "validation", "calibration", "test")
ARM_ALIASES = (
    ("known_marginal", "known_marginal"),
    ("empirical_stationary", "empirical_stationary"),
    ("ewma", "ewma"),
    ("logistic_ar32", "logistic_ar32"),
    ("parity_moment_ar", "parity_moment_ar"),
    ("grid_bayes", "grid_bayes"),
    ("latent_history_oracle", "latent_history_oracle"),
    ("contemporaneous_oracle", "contemporaneous_oracle"),
)


def _strict_object(value: object, cls: type[Any], label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")  # noqa: TRY004
    expected = {field.name for field in fields(cls)}
    actual = set(value)
    if missing := expected - actual:
        raise ValueError(f"missing fields in {label}: {sorted(missing)}")
    if unknown := actual - expected:
        raise ValueError(f"unknown fields in {label}: {sorted(unknown)}")
    return dict(value)


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _fixed(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} must equal the canonical fixed value {expected!r}")


def _fixed_int(actual: object, expected: int, label: str) -> None:
    _fixed(_integer(actual, label), expected, label)


def _fixed_number(actual: object, expected: float, label: str) -> None:
    _fixed(_number(actual, label), expected, label)


@dataclass(frozen=True, slots=True)
class CodeConfig:
    name: str
    ell: int
    n: int
    k: int
    distance_upper_bound: int
    hx_sha256: str
    hz_sha256: str

    def validate(self) -> None:
        expected = (
            "lp_3_7_16",
            45,
            2610,
            744,
            16,
            "fc685627e7a7139b6af9c12187879a02f19c46598fe99a9be816f58d4627ead8",
            "82b536419e91a3c877685d6ab347810878a46dfbff76373d5a02b160bb8d2ecb",
        )
        actual = (
            self.name,
            _integer(self.ell, "code.ell"),
            _integer(self.n, "code.n"),
            _integer(self.k, "code.k"),
            _integer(self.distance_upper_bound, "code.distance_upper_bound"),
            self.hx_sha256,
            self.hz_sha256,
        )
        if actual != expected:
            raise ValueError("code metadata must equal the canonical lp_3_7_16 identity")


@dataclass(frozen=True, slots=True)
class SplitConfig:
    train: int
    validation: int
    calibration: int
    test: int

    def validate(self) -> None:
        for name, expected in (("train", 8), ("validation", 8), ("calibration", 8), ("test", 64)):
            _fixed_int(getattr(self, name), expected, f"splits.{name}")


@dataclass(frozen=True, slots=True)
class RoundConfig:
    burn_in: int
    scored: int

    def validate(self) -> None:
        _fixed_int(self.burn_in, 64, "rounds.burn_in")
        _fixed_int(self.scored, 128, "rounds.scored")


@dataclass(frozen=True, slots=True)
class DynamicsConfig:
    base_probability: float
    ar_coefficient: float
    innovation_std: float
    clip: float
    probability_clip: tuple[float, float]

    def validate(self) -> None:
        _fixed_number(self.base_probability, 0.0375, "dynamics.base_probability")
        _fixed_number(self.ar_coefficient, 0.97, "dynamics.ar_coefficient")
        _fixed_number(self.innovation_std, 0.08, "dynamics.innovation_std")
        _fixed_number(self.clip, 1.2, "dynamics.clip")
        _fixed(self.probability_clip, (1e-5, 0.25), "dynamics.probability_clip")


@dataclass(frozen=True, slots=True)
class GridConfig:
    interior_cells: int
    doubled_interior_cells: int
    open_loop_interior_cells: int
    convergence_tolerance: float

    def validate(self) -> None:
        _fixed_int(self.interior_cells, 2048, "grid.interior_cells")
        _fixed_int(self.doubled_interior_cells, 4096, "grid.doubled_interior_cells")
        _fixed_int(self.open_loop_interior_cells, 4096, "grid.open_loop_interior_cells")
        _fixed_number(self.convergence_tolerance, 2.5e-5, "grid.convergence_tolerance")


@dataclass(frozen=True, slots=True)
class FisherConfig:
    draws: int
    draw_law: str
    finite_difference_step: float
    absolute_tolerance: float
    relative_tolerance: float

    def validate(self) -> None:
        _fixed_int(self.draws, 10_000, "fisher.draws")
        _fixed(self.draw_law, "stationary_normal_then_clip", "fisher.draw_law")
        _fixed_number(self.finite_difference_step, 1e-6, "fisher.finite_difference_step")
        _fixed_number(self.absolute_tolerance, 1e-8, "fisher.absolute_tolerance")
        _fixed_number(self.relative_tolerance, 1e-6, "fisher.relative_tolerance")


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    delta_nll: float
    bootstrap_draws: int
    one_sided_alpha: float
    two_sided_lower_quantile: float
    two_sided_upper_quantile: float
    holm_alpha: float
    nmse_denominator: float
    calibration_bins: int
    calibration_range: tuple[float, float]

    def validate(self) -> None:
        fixed_numbers = (
            ("delta_nll", 0.00025),
            ("one_sided_alpha", 0.05),
            ("two_sided_lower_quantile", 0.025),
            ("two_sided_upper_quantile", 0.975),
            ("holm_alpha", 0.05),
            ("nmse_denominator", 0.08**2 / (1 - 0.97**2)),
        )
        for name, expected in fixed_numbers:
            _fixed_number(getattr(self, name), expected, f"inference.{name}")
        _fixed_int(self.bootstrap_draws, 10_000, "inference.bootstrap_draws")
        _fixed_int(self.calibration_bins, 10, "inference.calibration_bins")
        _fixed(self.calibration_range, (1e-5, 0.25), "inference.calibration_range")


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    empirical_stationary_shrinkage: tuple[float, ...]
    ewma_decays: tuple[float, ...]
    ewma_kernel: int
    logistic_lags: int
    logistic_kernel: int
    logistic_l2: tuple[float, ...]
    lbfgs_max_iter: int
    tie_rule: str
    tie_tolerance: float
    calibration: str
    arm_aliases: tuple[tuple[str, str], ...]

    def validate(self) -> None:
        if type(self.empirical_stationary_shrinkage) is not tuple:
            raise ValueError("baselines.empirical_stationary_shrinkage must be an immutable tuple")
        shrinkage = tuple(
            _number(value, f"baselines.empirical_stationary_shrinkage[{index}]")
            for index, value in enumerate(self.empirical_stationary_shrinkage)
        )
        _fixed(shrinkage, (1.0,), "baselines.empirical_stationary_shrinkage")
        _fixed(self.ewma_decays, (0.5, 0.8, 0.9, 0.97, 0.99), "baselines.ewma_decays")
        _fixed_int(self.ewma_kernel, 5, "baselines.ewma_kernel")
        _fixed_int(self.logistic_lags, 32, "baselines.logistic_lags")
        _fixed_int(self.logistic_kernel, 3, "baselines.logistic_kernel")
        _fixed(self.logistic_l2, (1e-4, 1e-3, 1e-2, 1e-1), "baselines.logistic_l2")
        _fixed_int(self.lbfgs_max_iter, 500, "baselines.lbfgs_max_iter")
        _fixed(self.tie_rule, "sorted_grid_first_minimum", "baselines.tie_rule")
        _fixed_number(self.tie_tolerance, 1e-12, "baselines.tie_tolerance")
        _fixed(self.calibration, "identity", "baselines.calibration")
        _fixed(self.arm_aliases, ARM_ALIASES, "baselines.arm_aliases")


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    max_iter: int
    bp_method: str
    schedule: str
    ms_scaling_factor: float
    lsd_method: str
    lsd_order: int

    def validate(self) -> None:
        actual = BPLSDConfig(
            max_iter=_integer(self.max_iter, "decoder.max_iter"),
            bp_method=self.bp_method,
            schedule=self.schedule,
            ms_scaling_factor=_number(self.ms_scaling_factor, "decoder.ms_scaling_factor"),
            lsd_method=self.lsd_method,
            lsd_order=_integer(self.lsd_order, "decoder.lsd_order"),
        )
        if actual != BPLSDConfig():
            raise ValueError("decoder must equal the canonical fixed BP-LSD configuration")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    process_cpu_seconds: int

    def validate(self) -> None:
        _fixed_int(self.process_cpu_seconds, 21_600, "runtime.process_cpu_seconds")


@dataclass(frozen=True, slots=True)
class SeedConfig:
    campaign_domain: str
    campaign: int
    bootstrap_domain: str
    bootstrap: int
    derangement_domain: str
    derangement: int
    fisher_domain: str
    fisher: int

    def validate(self) -> None:
        expected = {
            "campaign": ("qldpc-fno/temporal-identifiability/v1", 7732479637849421559),
            "bootstrap": (
                "qldpc-fno/temporal-identifiability/bootstrap/v1",
                16303265125886503477,
            ),
            "derangement": (
                "qldpc-fno/temporal-identifiability/derangement/v1",
                13987031144127066471,
            ),
            "fisher": (
                "qldpc-fno/temporal-identifiability/fisher/v1",
                12048901516626741672,
            ),
        }
        for purpose, (domain, seed) in expected.items():
            _fixed(getattr(self, f"{purpose}_domain"), domain, f"seeds.{purpose}_domain")
            _fixed_int(getattr(self, purpose), seed, f"seeds.{purpose}")
            derived = int.from_bytes(hashlib.sha256(domain.encode()).digest()[:8], "big")
            _fixed(derived, seed, f"seeds.{purpose} SHA-256 derivation")


@dataclass(frozen=True, slots=True)
class IdentifiabilityConfig:
    schema_version: int
    artifact_label: str
    regimes: tuple[str, ...]
    code: CodeConfig
    splits: SplitConfig
    rounds: RoundConfig
    dynamics: DynamicsConfig
    grid: GridConfig
    fisher: FisherConfig
    inference: InferenceConfig
    baselines: BaselineConfig
    decoder: DecoderConfig
    runtime: RuntimeConfig
    seeds: SeedConfig

    @property
    def roles(self) -> tuple[str, ...]:
        return ROLES

    def validate(self) -> None:
        _fixed_int(self.schema_version, 1, "schema_version")
        _fixed(self.artifact_label, "confirmatory_gate", "artifact_label")
        if type(self.regimes) is not tuple or self.regimes != REGIMES:
            raise ValueError(f"regimes must equal the immutable canonical tuple {REGIMES!r}")
        for section in (
            self.code,
            self.splits,
            self.rounds,
            self.dynamics,
            self.grid,
            self.fisher,
            self.inference,
            self.baselines,
            self.decoder,
            self.runtime,
            self.seeds,
        ):
            section.validate()


_SECTIONS: tuple[tuple[str, type[Any]], ...] = (
    ("code", CodeConfig),
    ("splits", SplitConfig),
    ("rounds", RoundConfig),
    ("dynamics", DynamicsConfig),
    ("grid", GridConfig),
    ("fisher", FisherConfig),
    ("inference", InferenceConfig),
    ("baselines", BaselineConfig),
    ("decoder", DecoderConfig),
    ("runtime", RuntimeConfig),
    ("seeds", SeedConfig),
)


def load_identifiability_config(path: Path) -> IdentifiabilityConfig:
    """Load only the immutable, canonical scientific configuration."""
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read identifiability config: {exc}") from exc
    top = _strict_object(raw, IdentifiabilityConfig, "configuration")
    for name, cls in _SECTIONS:
        values = _strict_object(top[name], cls, name)
        if name in {"dynamics", "inference"}:
            range_name = "probability_clip" if name == "dynamics" else "calibration_range"
            range_value = values[range_name]
            if not isinstance(range_value, list):
                raise ValueError(f"{name}.{range_name} must be a JSON array")
            values[range_name] = tuple(range_value)
        elif name == "baselines":
            for field_name in ("empirical_stationary_shrinkage", "ewma_decays", "logistic_l2"):
                value = values[field_name]
                if not isinstance(value, list):
                    raise ValueError(  # noqa: TRY004
                        f"baselines.{field_name} must be a JSON array"
                    )
                values[field_name] = tuple(value)
            aliases = values["arm_aliases"]
            if not isinstance(aliases, Mapping):
                raise ValueError("baselines.arm_aliases must be an object")
            values["arm_aliases"] = tuple(aliases.items())
        top[name] = cls(**values)
    regimes = top["regimes"]
    if not isinstance(regimes, list):
        raise ValueError("regimes must be a JSON array")  # noqa: TRY004
    top["regimes"] = tuple(regimes)
    config = IdentifiabilityConfig(**top)
    config.validate()
    return config
