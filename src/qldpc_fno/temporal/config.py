"""Strict immutable configuration for causal FNO-HiPPO experiments."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Self

from qldpc_fno.temporal.seeds import REGIMES


def _strict_section(value: object, section_type: type[Any], name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")  # noqa: TRY004
    expected = {field.name for field in fields(section_type)}
    actual = set(value)
    unknown = actual - expected
    missing = expected - actual
    if unknown:
        raise ValueError(f"unknown fields in {name}: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing fields in {name}: {sorted(missing)}")
    return dict(value)


def _integer(value: object, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, *, lower: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and numeric")
    result = float(value)
    if lower is not None and result <= lower:
        raise ValueError(f"{name} must be greater than {lower}")
    return result


@dataclass(frozen=True, slots=True)
class CodeIdentity:
    name: str
    ell: int
    n: int
    k: int
    distance_upper_bound: int

    def validate(self) -> None:
        if type(self.name) is not str or self.name != "lp_3_7_16":
            raise ValueError("code name must be 'lp_3_7_16'")
        _integer(self.ell, "ell")
        _integer(self.n, "n")
        _integer(self.k, "k")
        _integer(self.distance_upper_bound, "distance_upper_bound")
        if (self.ell, self.n, self.k, self.distance_upper_bound) != (45, 2610, 744, 16):
            raise ValueError("code must identify the canonical lp_3_7_16 geometry")


@dataclass(frozen=True, slots=True)
class SplitSizes:
    train: int
    validation: int
    calibration: int
    test: int

    def validate(self) -> None:
        _integer(self.train, "train")
        _integer(self.validation, "validation")
        _integer(self.calibration, "calibration")
        _integer(self.test, "test", minimum=0)


@dataclass(frozen=True, slots=True)
class RoundConfig:
    burn_in: int
    scored: int

    def validate(self) -> None:
        _integer(self.burn_in, "burn_in", minimum=0)
        _integer(self.scored, "scored")


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    base_probability: float
    min_probability: float
    max_probability: float
    channel_offset_std: float
    static_spatial_amplitude: float
    spatial_frequency_min: int
    spatial_frequency_max: int
    global_ar_coefficient: float
    global_innovation_std: float
    global_clip: float
    joint_base_amplitude: float
    burst_start_probability: float
    burst_amplitude: float
    burst_profile_concentration: float
    burst_decay_rounds: float
    burst_max_age: int
    burst_min_amplitude: float
    mismatch_duration_min: int
    mismatch_duration_max: int
    mismatch_width_min: int
    mismatch_width_max: int
    mismatch_step_min: int
    mismatch_step_max: int

    @property
    def probability_bounds(self) -> tuple[float, float]:
        return (self.min_probability, self.max_probability)

    @property
    def spatial_frequency_range(self) -> tuple[int, int]:
        return (self.spatial_frequency_min, self.spatial_frequency_max)

    @property
    def mismatch_step_range(self) -> tuple[int, int]:
        return (self.mismatch_step_min, self.mismatch_step_max)

    def validate(self) -> None:
        lower = _number(self.min_probability, "min_probability", lower=0.0)
        base = _number(self.base_probability, "base_probability", lower=0.0)
        upper = _number(self.max_probability, "max_probability", lower=0.0)
        if not lower < base < upper < 0.5:
            raise ValueError("probability bounds must satisfy 0 < min < base < max < 0.5")
        _number(self.channel_offset_std, "channel_offset_std", lower=0.0)
        _number(self.static_spatial_amplitude, "static_spatial_amplitude", lower=0.0)
        frequency_min = _integer(self.spatial_frequency_min, "spatial_frequency_min")
        frequency_max = _integer(self.spatial_frequency_max, "spatial_frequency_max")
        if frequency_min > frequency_max:
            raise ValueError("spatial frequency range is reversed")
        ar = _number(self.global_ar_coefficient, "global_ar_coefficient")
        if not 0.0 <= ar < 1.0:
            raise ValueError("global_ar_coefficient must be in [0, 1)")
        _number(self.global_innovation_std, "global_innovation_std", lower=0.0)
        _number(self.global_clip, "global_clip", lower=0.0)
        _number(self.joint_base_amplitude, "joint_base_amplitude", lower=0.0)
        start = _number(self.burst_start_probability, "burst_start_probability")
        if not 0.0 <= start <= 1.0:
            raise ValueError("burst_start_probability must be in [0, 1]")
        _number(self.burst_amplitude, "burst_amplitude", lower=0.0)
        _number(self.burst_profile_concentration, "burst_profile_concentration", lower=0.0)
        _number(self.burst_decay_rounds, "burst_decay_rounds", lower=0.0)
        _integer(self.burst_max_age, "burst_max_age", minimum=0)
        _number(self.burst_min_amplitude, "burst_min_amplitude", lower=0.0)
        duration_min = _integer(self.mismatch_duration_min, "mismatch_duration_min")
        duration_max = _integer(self.mismatch_duration_max, "mismatch_duration_max")
        width_min = _integer(self.mismatch_width_min, "mismatch_width_min")
        width_max = _integer(self.mismatch_width_max, "mismatch_width_max")
        if duration_min > duration_max:
            raise ValueError("mismatch duration range is reversed")
        if width_min > width_max:
            raise ValueError("mismatch width range is reversed")
        if type(self.mismatch_step_min) is not int or type(self.mismatch_step_max) is not int:
            raise ValueError("mismatch steps must be integers")
        step_min = self.mismatch_step_min
        step_max = self.mismatch_step_max
        if step_min < -1 or step_max > 1:
            raise ValueError("mismatch steps must be within [-1, 1]")
        if step_min > step_max:
            raise ValueError("mismatch step range is reversed")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    hidden_width: int
    fno_modes: int
    fir_history: int
    hippo_order: int
    gru_state_width: int

    def validate(self, ell: int) -> None:
        _integer(self.hidden_width, "hidden_width")
        modes = _integer(self.fno_modes, "fno_modes")
        if modes > ell // 2 + 1:
            raise ValueError("fno_modes exceeds the real FFT mode count")
        _integer(self.fir_history, "fir_history")
        _integer(self.hippo_order, "hippo_order")
        _integer(self.gru_state_width, "gru_state_width")


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    max_iter: int
    bp_method: str
    schedule: str
    ms_scaling_factor: float
    lsd_method: str
    lsd_order: int

    def validate(self) -> None:
        _integer(self.max_iter, "max_iter")
        if self.bp_method != "minimum_sum":
            raise ValueError("bp_method must be 'minimum_sum'")
        if self.schedule != "serial":
            raise ValueError("schedule must be 'serial'")
        scaling = _number(self.ms_scaling_factor, "ms_scaling_factor")
        if scaling < 0.0:
            raise ValueError("ms_scaling_factor must be non-negative")
        if self.lsd_method != "LSD_E":
            raise ValueError("lsd_method must be 'LSD_E'")
        _integer(self.lsd_order, "lsd_order", minimum=0)


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    gradient_norm_cap: float
    training_seed: int

    def validate(self) -> None:
        _number(self.learning_rate, "learning_rate", lower=0.0)
        weight_decay = _number(self.weight_decay, "weight_decay")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        _integer(self.batch_size, "batch_size")
        _integer(self.max_epochs, "max_epochs")
        _number(self.gradient_norm_cap, "gradient_norm_cap", lower=0.0)
        _integer(self.training_seed, "training_seed", minimum=0)


@dataclass(frozen=True, slots=True)
class CausalExperimentConfig:
    """Complete causal experiment policy with recursively strict JSON parsing."""

    campaign_seed: int
    artifact_mode: str
    regimes: tuple[str, ...]
    code: CodeIdentity
    splits: SplitSizes
    rounds: RoundConfig
    generator: GeneratorConfig
    model: ModelConfig
    decoder: DecoderConfig
    optimizer: OptimizerConfig

    _ARTIFACT_MODES: ClassVar[frozenset[str]] = frozenset(
        {"reduced_non_scientific", "discovery_non_scientific", "confirmation"}
    )

    @classmethod
    def from_json(cls, path: Path) -> Self:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"unable to read causal experiment configuration: {path}") from error
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: object) -> Self:
        values = _strict_section(payload, cls, "configuration")
        regimes = values["regimes"]
        if not isinstance(regimes, Sequence) or isinstance(regimes, (str, bytes)):
            raise ValueError("regimes must contain the five named regimes")  # noqa: TRY004
        config = cls(
            campaign_seed=values["campaign_seed"],
            artifact_mode=values["artifact_mode"],
            regimes=tuple(regimes),
            code=CodeIdentity(**_strict_section(values["code"], CodeIdentity, "code")),
            splits=SplitSizes(**_strict_section(values["splits"], SplitSizes, "splits")),
            rounds=RoundConfig(**_strict_section(values["rounds"], RoundConfig, "rounds")),
            generator=GeneratorConfig(
                **_strict_section(values["generator"], GeneratorConfig, "generator")
            ),
            model=ModelConfig(**_strict_section(values["model"], ModelConfig, "model")),
            decoder=DecoderConfig(
                **_strict_section(values["decoder"], DecoderConfig, "decoder")
            ),
            optimizer=OptimizerConfig(
                **_strict_section(values["optimizer"], OptimizerConfig, "optimizer")
            ),
        )
        config.validate()
        return config

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def validate(self) -> None:
        _integer(self.campaign_seed, "campaign_seed", minimum=0)
        if not isinstance(self.artifact_mode, str) or self.artifact_mode not in self._ARTIFACT_MODES:
            raise ValueError(
                "artifact_mode must be reduced_non_scientific, discovery_non_scientific, "
                "or confirmation"
            )
        if self.regimes != REGIMES:
            raise ValueError("regimes must contain the five named regimes in canonical order")
        self.code.validate()
        self.splits.validate()
        self.rounds.validate()
        self.generator.validate()
        self.model.validate(self.code.ell)
        self.decoder.validate()
        self.optimizer.validate()
