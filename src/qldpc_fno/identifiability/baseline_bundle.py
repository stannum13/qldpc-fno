"""Frozen, safe evidence bundles for the preregistered baseline fits."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qldpc_fno.codes.lifted_product import build_self_lifted_product
from qldpc_fno.codes.seeds import PAPER_LP_3_7_16
from qldpc_fno.identifiability.config import (
    ARM_ALIASES,
    BaselineConfig,
    CodeConfig,
    DecoderConfig,
    DynamicsConfig,
    FisherConfig,
    GridConfig,
    IdentifiabilityConfig,
    InferenceConfig,
    RoundConfig,
    RuntimeConfig,
    SeedConfig,
    SplitConfig,
)
from qldpc_fno.identifiability.generator import generate_scalar_sequence
from qldpc_fno.identifiability.types import (
    DevelopmentPartitions,
    GeneratedSequence,
    SequenceIdentity,
)
from qldpc_fno.temporal.baselines import (
    CircularLogisticForecaster,
    StationaryForecaster,
    fit_ewma,
    fit_logistic_ar,
    fit_stationary,
)

_SCHEMA_VERSION = 1
_POLICY_VERSION = "qldpc-fno/identifiability/baseline-fit/v1"
_FITTED_ARMS = ("empirical_stationary", "ewma", "logistic_ar32")


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(np.array(value, dtype=np.float64, copy=True))
    result.setflags(write=False)
    return result


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _identity_payload(identity: SequenceIdentity) -> dict[str, object]:
    return dataclasses.asdict(identity)


def _partitions_payload(partitions: DevelopmentPartitions) -> dict[str, object]:
    return {
        role: [_identity_payload(identity) for identity in getattr(partitions, role)]
        for role in ("train", "validation", "calibration")
    }


def _validate_partitions(partitions: DevelopmentPartitions) -> DevelopmentPartitions:
    if type(partitions) is not DevelopmentPartitions:
        raise TypeError("baseline fitting requires exact DevelopmentPartitions")
    # Re-run construction because frozen dataclasses can be forged through low-level APIs.
    DevelopmentPartitions(partitions.train, partitions.validation, partitions.calibration)
    if not all((*partitions.train, *partitions.validation, *partitions.calibration)):
        raise ValueError("each development role requires content-bound identities")
    return partitions


def _config_payload(config: IdentifiabilityConfig) -> dict[str, object]:
    return dataclasses.asdict(config)


def _config_from_payload(value: object) -> IdentifiabilityConfig:
    if not isinstance(value, dict):
        raise ValueError("bundle config payload must be an object")  # noqa: TRY004
    try:
        config = IdentifiabilityConfig(
            schema_version=value["schema_version"],
            artifact_label=value["artifact_label"],
            regimes=tuple(value["regimes"]),
            code=CodeConfig(**value["code"]),
            splits=SplitConfig(**value["splits"]),
            rounds=RoundConfig(**value["rounds"]),
            dynamics=DynamicsConfig(
                **{**value["dynamics"], "probability_clip": tuple(value["dynamics"]["probability_clip"])}
            ),
            grid=GridConfig(**value["grid"]),
            fisher=FisherConfig(**value["fisher"]),
            inference=InferenceConfig(
                **{**value["inference"], "calibration_range": tuple(value["inference"]["calibration_range"])}
            ),
            baselines=BaselineConfig(
                **{
                    **value["baselines"],
                    "empirical_stationary_shrinkage": tuple(value["baselines"]["empirical_stationary_shrinkage"]),
                    "ewma_decays": tuple(value["baselines"]["ewma_decays"]),
                    "logistic_l2": tuple(value["baselines"]["logistic_l2"]),
                    "arm_aliases": tuple(tuple(item) for item in value["baselines"]["arm_aliases"]),
                }
            ),
            decoder=DecoderConfig(**value["decoder"]),
            runtime=RuntimeConfig(**value["runtime"]),
            seeds=SeedConfig(**value["seeds"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("bundle config payload is incomplete") from exc
    config.validate()
    return config


def _identities_from_payload(value: object) -> tuple[SequenceIdentity, ...]:
    if not isinstance(value, list):
        raise ValueError("bundle identity list must be an array")  # noqa: TRY004
    try:
        return tuple(SequenceIdentity(**item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("bundle identity payload is invalid") from exc


def _partitions_from_payload(value: object) -> DevelopmentPartitions:
    if not isinstance(value, dict) or set(value) != {"train", "validation", "calibration"}:
        raise ValueError("bundle partition payload is invalid")
    return DevelopmentPartitions(
        _identities_from_payload(value["train"]),
        _identities_from_payload(value["validation"]),
        _identities_from_payload(value["calibration"]),
    )


@dataclass(frozen=True, slots=True)
class FrozenEstimator:
    """One exact baseline state with identity calibration only."""

    name: str
    predictor_type: str
    scalar_probability: float | None
    shrinkage: float | None
    l2: float | None
    feature_kind: str | None
    decay: float | None
    lags: int | None
    temperature: float
    empirical_field: np.ndarray
    weight: np.ndarray
    bias: np.ndarray

    def __post_init__(self) -> None:
        if self.name not in _FITTED_ARMS or self.temperature != 1.0:
            raise ValueError("frozen estimators require a canonical name and identity calibration")
        if self.predictor_type not in {"stationary", "circular_logistic"}:
            raise ValueError("frozen estimator type is not registered")
        object.__setattr__(self, "empirical_field", _readonly(self.empirical_field))
        object.__setattr__(self, "weight", _readonly(self.weight))
        object.__setattr__(self, "bias", _readonly(self.bias))

    def materialize(self) -> StationaryForecaster | CircularLogisticForecaster:
        if self.predictor_type == "stationary":
            if self.scalar_probability is None or self.shrinkage is None:
                raise ValueError("stationary frozen estimator is incomplete")
            return StationaryForecaster(
                self.scalar_probability, self.empirical_field, self.shrinkage, self.temperature
            )
        if self.l2 is None or self.feature_kind is None or self.lags is None:
            raise ValueError("logistic frozen estimator is incomplete")
        return CircularLogisticForecaster(
            self.weight.copy(), self.bias.copy(), self.l2, self.temperature, self.feature_kind, self.decay, self.lags
        )


def _arm_payload(arm: FrozenEstimator) -> dict[str, object]:
    arrays = {
        "empirical_field": arm.empirical_field,
        "weight": arm.weight,
        "bias": arm.bias,
    }
    return {
        "name": arm.name,
        "predictor_type": arm.predictor_type,
        "scalar_probability": arm.scalar_probability,
        "shrinkage": arm.shrinkage,
        "l2": arm.l2,
        "feature_kind": arm.feature_kind,
        "decay": arm.decay,
        "lags": arm.lags,
        "temperature": arm.temperature,
        "arrays": {
            name: {"dtype": value.dtype.str, "shape": list(value.shape), "sha256": _array_digest(value)}
            for name, value in arrays.items()
        },
    }


def _bundle_payload(bundle: FrozenEstimatorBundle) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "policy_version": _POLICY_VERSION,
        "config": bundle.config_payload,
        "config_sha256": bundle.config_sha256,
        "partitions": _partitions_payload(bundle.partitions),
        "partition_sha256": _digest(_partitions_payload(bundle.partitions)),
        "policy_sha256": bundle.policy_sha256,
        "arm_aliases": [list(alias) for alias in bundle.arm_aliases],
        "arms": [_arm_payload(arm) for arm in bundle.arms],
    }


@dataclass(frozen=True, slots=True)
class FrozenEstimatorBundle:
    arms: tuple[FrozenEstimator, ...]
    arm_aliases: tuple[tuple[str, str], ...]
    partitions: DevelopmentPartitions
    config_payload: dict[str, object]
    config_sha256: str
    policy_sha256: str
    integrity_sha256: str

    def __post_init__(self) -> None:
        if type(self.arms) is not tuple or tuple(arm.name for arm in self.arms) != _FITTED_ARMS:
            raise ValueError("bundle arms must use the canonical fitted-arm order")
        if any(type(arm) is not FrozenEstimator for arm in self.arms):
            raise TypeError("bundle requires exact frozen estimator states")
        _validate_partitions(self.partitions)
        if self.arm_aliases != ARM_ALIASES:
            raise ValueError("bundle arm aliases must equal the canonical aliases")
        config = _config_from_payload(self.config_payload)
        if self.config_sha256 != _digest(self.config_payload):
            raise ValueError("bundle config hash disagrees with its payload")
        if config.baselines.arm_aliases != self.arm_aliases:
            raise ValueError("bundle alias policy disagrees with its configuration")
        expected_policy = _policy_hash(config)
        if self.policy_sha256 != expected_policy:
            raise ValueError("bundle baseline policy hash disagrees with configuration")
        expected_integrity = _digest(_bundle_payload_without_integrity(self))
        if self.integrity_sha256 != expected_integrity:
            raise ValueError("bundle integrity hash disagrees with frozen parameters")

    def arm(self, name: str) -> FrozenEstimator:
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise KeyError(name)


def _bundle_payload_without_integrity(bundle: FrozenEstimatorBundle) -> dict[str, object]:
    return _bundle_payload(bundle)


@dataclass(frozen=True, slots=True)
class BundleManifest:
    schema_version: int
    metadata_sha256: str
    arrays_sha256: str
    integrity_sha256: str
    array_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("bundle manifest schema version is unsupported")


def _policy_hash(config: IdentifiabilityConfig) -> str:
    policy = {"version": _POLICY_VERSION, **dataclasses.asdict(config.baselines)}
    return _digest(policy)


def _generate(identity: SequenceIdentity, config: IdentifiabilityConfig) -> GeneratedSequence:
    code = build_self_lifted_product(PAPER_LP_3_7_16)
    sequence = generate_scalar_sequence(
        config,
        regime=identity.regime,
        role=identity.role,
        sequence_index=identity.sequence_index,
        code=code,
    )
    if sequence.identity != identity:
        raise ValueError("regenerated development content does not match its frozen identity")
    return sequence


def _syndrome_to_qubit_features(syndromes: np.ndarray, target_width: int) -> np.ndarray:
    """Losslessly present raw public and target rings to the circular fitters."""
    observed = np.asarray(syndromes, dtype=np.uint8)
    if observed.ndim != 4 or observed.shape[2] != 1:
        raise ValueError("raw baseline syndromes must have one public channel")
    if observed.shape[-1] != 945 or target_width != 2610:
        raise ValueError("baseline geometry must equal canonical 945-check and 2610-qubit rings")
    return observed.reshape(*observed.shape[:2], 21, 45).astype(np.float64)


def _role_data(
    identities: tuple[SequenceIdentity, ...], config: IdentifiabilityConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequences = tuple(_generate(identity, config) for identity in identities)
    if not sequences:
        raise ValueError("baseline fitting requires at least one sequence in every development role")
    if any(sequence.identity != identity for sequence, identity in zip(sequences, identities, strict=True)):
        raise ValueError("generated development content does not match the frozen role identity")
    syndromes = np.stack([sequence.deployable.syndromes for sequence in sequences])[:, :, None, :]
    targets = np.stack([sequence.targets.errors for sequence in sequences])
    masks = np.stack([sequence.deployable.scored_mask for sequence in sequences])
    if not masks.any() or not np.all(masks.any(axis=1)):
        raise ValueError("development roles require scored rounds")
    features = _syndrome_to_qubit_features(syndromes, targets.shape[-1])
    return features, targets.reshape(*targets.shape[:2], 58, 45).astype(np.float64), masks.astype(np.bool_)


@contextmanager
def _deterministic_torch() -> Any:
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    threads = torch.get_num_threads()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(was_deterministic)


def _freeze(name: str, model: object) -> FrozenEstimator:
    if type(model) is StationaryForecaster:
        if model.shrinkage != 1.0 or model.temperature != 1.0:
            raise ValueError("empirical stationary baseline must be raw with identity calibration")
        return FrozenEstimator(
            name, "stationary", float(model.scalar_probability), float(model.shrinkage), None, None, None, None,
            1.0, model.empirical_field, np.empty(0), np.empty(0)
        )
    if type(model) is not CircularLogisticForecaster or model.temperature != 1.0:
        raise TypeError("baseline fitter returned an invalid non-identity deployable estimator")
    expected = "ewma" if name == "ewma" else "lagged"
    if model.feature_kind != expected:
        raise ValueError("baseline fitter returned a model in the wrong canonical slot")
    return FrozenEstimator(
        name, "circular_logistic", None, None, float(model.l2), model.feature_kind,
        None if model.decay is None else float(model.decay), int(model.lags), 1.0,
        np.empty(0), model.weight, model.bias,
    )


def fit_development_bundle(
    partitions: DevelopmentPartitions, config: IdentifiabilityConfig
) -> FrozenEstimatorBundle:
    """Fit only regenerated development roles and freeze exact raw estimator state."""
    partitions = _validate_partitions(partitions)
    if type(config) is not IdentifiabilityConfig:
        raise TypeError("baseline fitting requires exact IdentifiabilityConfig")
    config.validate()
    train_x, train_y, train_mask = _role_data(partitions.train, config)
    validation_x, validation_y, validation_mask = _role_data(partitions.validation, config)
    calibration_x, calibration_y, calibration_mask = _role_data(partitions.calibration, config)
    # Touch calibration data only to validate its role and scored mask: this
    # preregistration requires the identity transform, so it cannot tune state.
    del calibration_x, calibration_y
    if not calibration_mask.any():
        raise ValueError("calibration role must contain scored rounds")
    with _deterministic_torch():
        stationary = fit_stationary(
            train_y, validation_y, train_mask=train_mask, validation_mask=validation_mask,
            lambda_grid=config.baselines.empirical_stationary_shrinkage, calibrate=None,
        )
        ewma = fit_ewma(
            train_x, train_y, validation_x, validation_y, train_mask=train_mask, validation_mask=validation_mask,
            decays=config.baselines.ewma_decays, kernel_size=config.baselines.ewma_kernel,
            l2_grid=config.baselines.logistic_l2, max_iter=config.baselines.lbfgs_max_iter, calibrate=None,
        )
        logistic = fit_logistic_ar(
            train_x, train_y, validation_x, validation_y, train_mask=train_mask, validation_mask=validation_mask,
            lags=config.baselines.logistic_lags, kernel_size=config.baselines.logistic_kernel,
            l2_grid=config.baselines.logistic_l2, max_iter=config.baselines.lbfgs_max_iter, calibrate=None,
        )
    arms = (_freeze("empirical_stationary", stationary), _freeze("ewma", ewma), _freeze("logistic_ar32", logistic))
    config_payload = _config_payload(config)
    preliminary = FrozenEstimatorBundle.__new__(FrozenEstimatorBundle)
    object.__setattr__(preliminary, "arms", arms)
    object.__setattr__(preliminary, "arm_aliases", config.baselines.arm_aliases)
    object.__setattr__(preliminary, "partitions", partitions)
    object.__setattr__(preliminary, "config_payload", config_payload)
    object.__setattr__(preliminary, "config_sha256", _digest(config_payload))
    object.__setattr__(preliminary, "policy_sha256", _policy_hash(config))
    object.__setattr__(preliminary, "integrity_sha256", "")
    integrity = _digest(_bundle_payload_without_integrity(preliminary))
    return FrozenEstimatorBundle(
        arms, config.baselines.arm_aliases, partitions, config_payload, _digest(config_payload), _policy_hash(config), integrity
    )


def _array_file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_frozen_bundle(path: Path, bundle: FrozenEstimatorBundle) -> BundleManifest:
    """Persist only JSON plus numeric NPY members; executable pickle is forbidden."""
    if not isinstance(path, Path) or type(bundle) is not FrozenEstimatorBundle:
        raise TypeError("bundle path and bundle must have valid public types")
    # Revalidate integrity before publication.
    FrozenEstimatorBundle(
        bundle.arms, bundle.arm_aliases, bundle.partitions, bundle.config_payload,
        bundle.config_sha256, bundle.policy_sha256, bundle.integrity_sha256,
    )
    if path.exists():
        raise ValueError("refusing to overwrite an existing frozen bundle")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    arrays: dict[str, np.ndarray] = {}
    for arm in bundle.arms:
        for name, value in (("empirical_field", arm.empirical_field), ("weight", arm.weight), ("bias", arm.bias)):
            arrays[f"{arm.name}__{name}"] = np.ascontiguousarray(value)
    try:
        array_path = temporary / "arrays.npz"
        np.savez(array_path, **arrays)
        metadata = _bundle_payload(bundle)
        metadata["integrity_sha256"] = bundle.integrity_sha256
        metadata_path = temporary / "metadata.json"
        metadata_path.write_bytes(_json_bytes(metadata))
        manifest = BundleManifest(_SCHEMA_VERSION, _array_file_digest(metadata_path), _array_file_digest(array_path), bundle.integrity_sha256, tuple(arrays))
        os.replace(temporary, path)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _read_arrays(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            if tuple(loaded.files) != names:
                raise ValueError("safe bundle array names are missing, renamed, or reordered")
            arrays = {name: np.array(loaded[name], copy=True) for name in names}
    except (OSError, ValueError) as exc:
        raise ValueError("failed to read safe non-pickle bundle arrays") from exc
    if any(array.dtype == object or not np.all(np.isfinite(array)) for array in arrays.values()):
        raise ValueError("safe bundle arrays must be finite numeric values")
    return arrays


def read_verified_bundle(path: Path, manifest: BundleManifest) -> FrozenEstimatorBundle:
    """Load, hash-check, and deterministically replay a development-only bundle."""
    if not isinstance(path, Path) or type(manifest) is not BundleManifest:
        raise TypeError("bundle path and manifest must have valid public types")
    if manifest.schema_version != _SCHEMA_VERSION:
        raise ValueError("bundle manifest schema version is unsupported")
    metadata_path, array_path = path / "metadata.json", path / "arrays.npz"
    if not metadata_path.is_file() or not array_path.is_file():
        raise ValueError("frozen bundle is missing metadata or arrays")
    if _array_file_digest(metadata_path) != manifest.metadata_sha256 or _array_file_digest(array_path) != manifest.arrays_sha256:
        raise ValueError("frozen bundle metadata or arrays were tampered")
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("frozen bundle metadata is invalid") from exc
    required = {
        "schema_version", "policy_version", "config", "config_sha256", "partitions", "partition_sha256", "policy_sha256", "arm_aliases", "arms", "integrity_sha256"
    }
    if not isinstance(metadata, dict) or set(metadata) != required or metadata["schema_version"] != _SCHEMA_VERSION or metadata["policy_version"] != _POLICY_VERSION:
        raise ValueError("frozen bundle metadata schema is invalid")
    if metadata["integrity_sha256"] != manifest.integrity_sha256:
        raise ValueError("manifest integrity does not match bundle metadata")
    if not isinstance(metadata["arms"], list) or len(metadata["arms"]) != len(_FITTED_ARMS):
        raise ValueError("frozen bundle arm metadata is incomplete")
    names = tuple(f"{arm}__{array}" for arm in _FITTED_ARMS for array in ("empirical_field", "weight", "bias"))
    if manifest.array_names != names:
        raise ValueError("manifest safe-array layout is not canonical")
    arrays = _read_arrays(array_path, names)
    arms: list[FrozenEstimator] = []
    for expected_name, item in zip(_FITTED_ARMS, metadata["arms"], strict=True):
        if not isinstance(item, dict) or item.get("name") != expected_name or set(item) != {
            "name", "predictor_type", "scalar_probability", "shrinkage", "l2", "feature_kind", "decay", "lags", "temperature", "arrays"
        }:
            raise ValueError("frozen bundle arm metadata is invalid or renamed")
        descriptors = item["arrays"]
        if not isinstance(descriptors, dict) or set(descriptors) != {"empirical_field", "weight", "bias"}:
            raise ValueError("frozen bundle array descriptors are invalid")
        for array_name, descriptor in descriptors.items():
            value = arrays[f"{expected_name}__{array_name}"]
            if not isinstance(descriptor, dict) or descriptor != {"dtype": value.dtype.str, "shape": list(value.shape), "sha256": _array_digest(value)}:
                raise ValueError("frozen bundle parameters were rehashed or tampered")
        arms.append(FrozenEstimator(
            item["name"], item["predictor_type"], item["scalar_probability"], item["shrinkage"], item["l2"], item["feature_kind"], item["decay"], item["lags"], item["temperature"],
            arrays[f"{expected_name}__empirical_field"], arrays[f"{expected_name}__weight"], arrays[f"{expected_name}__bias"],
        ))
    aliases = tuple(tuple(item) for item in metadata["arm_aliases"])
    bundle = FrozenEstimatorBundle(
        tuple(arms), aliases, _partitions_from_payload(metadata["partitions"]), metadata["config"], metadata["config_sha256"], metadata["policy_sha256"], metadata["integrity_sha256"],
    )
    if _digest(_partitions_payload(bundle.partitions)) != metadata["partition_sha256"]:
        raise ValueError("frozen bundle partition identities were rehashed or tampered")
    replay = fit_development_bundle(bundle.partitions, _config_from_payload(bundle.config_payload))
    if replay.integrity_sha256 != bundle.integrity_sha256:
        raise ValueError("deterministic development replay disagrees with frozen estimator parameters")
    return bundle


__all__ = [
    "BundleManifest",
    "FrozenEstimator",
    "FrozenEstimatorBundle",
    "fit_development_bundle",
    "read_verified_bundle",
    "write_frozen_bundle",
]
