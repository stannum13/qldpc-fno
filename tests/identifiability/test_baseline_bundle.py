from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from qldpc_fno.identifiability.baseline_bundle import (
    BundleManifest,
    FrozenEstimatorBundle,
    _syndrome_to_qubit_features,
    fit_development_bundle,
    read_verified_bundle,
    write_frozen_bundle,
)
from qldpc_fno.identifiability.config import load_identifiability_config
from qldpc_fno.identifiability.types import DevelopmentPartitions, SequenceIdentity


def _identity(role: str, index: int, character: str) -> SequenceIdentity:
    return SequenceIdentity(
        regime="stationary_iid",
        role=role,
        sequence_index=index,
        latent_seed=index + 101,
        bernoulli_seed=index + 201,
        content_sha256=character * 64,
    )


@pytest.fixture
def partitions() -> DevelopmentPartitions:
    return DevelopmentPartitions(
        train=(_identity("train", 0, "a"),),
        validation=(_identity("validation", 0, "b"),),
        calibration=(_identity("calibration", 0, "c"),),
    )


def test_fit_freezes_exact_policy_and_identity_calibration(
    monkeypatch: pytest.MonkeyPatch, partitions: DevelopmentPartitions
) -> None:
    config = load_identifiability_config(Path("configs/temporal_identifiability.json"))
    calls: dict[str, dict[str, object]] = {}

    def stationary(*args, **kwargs):
        calls["stationary"] = {"args": args, **kwargs}
        from qldpc_fno.temporal.baselines import StationaryForecaster

        return StationaryForecaster(0.1, np.array([[0.1]]), 1.0)

    def ewma(*args, **kwargs):
        calls["ewma"] = {"args": args, **kwargs}
        from qldpc_fno.temporal.baselines import CircularLogisticForecaster

        return CircularLogisticForecaster(np.zeros((1, 1, 5)), np.zeros(1), 1e-4, feature_kind="ewma", decay=0.5)

    def logistic(*args, **kwargs):
        calls["logistic"] = {"args": args, **kwargs}
        from qldpc_fno.temporal.baselines import CircularLogisticForecaster

        return CircularLogisticForecaster(np.zeros((1, 32, 3)), np.zeros(1), 1e-4, feature_kind="lagged", lags=32)

    monkeypatch.setattr("qldpc_fno.identifiability.baseline_bundle._generate", lambda identity, config: _sequence(identity))
    monkeypatch.setattr("qldpc_fno.identifiability.baseline_bundle.fit_stationary", stationary)
    monkeypatch.setattr("qldpc_fno.identifiability.baseline_bundle.fit_ewma", ewma)
    monkeypatch.setattr("qldpc_fno.identifiability.baseline_bundle.fit_logistic_ar", logistic)

    bundle = fit_development_bundle(partitions, config)

    assert isinstance(bundle, FrozenEstimatorBundle)
    assert calls["stationary"]["lambda_grid"] == config.baselines.empirical_stationary_shrinkage
    assert calls["ewma"]["decays"] == config.baselines.ewma_decays
    assert calls["ewma"]["kernel_size"] == config.baselines.ewma_kernel
    assert calls["ewma"]["l2_grid"] == config.baselines.logistic_l2
    assert calls["ewma"]["max_iter"] == config.baselines.lbfgs_max_iter
    assert calls["ewma"]["calibrate"] is None
    assert calls["logistic"]["lags"] == config.baselines.logistic_lags
    assert calls["logistic"]["kernel_size"] == config.baselines.logistic_kernel
    assert calls["logistic"]["l2_grid"] == config.baselines.logistic_l2
    assert calls["logistic"]["max_iter"] == config.baselines.lbfgs_max_iter
    assert calls["logistic"]["calibrate"] is None
    assert calls["stationary"]["calibrate"] is None
    assert config.baselines.tie_rule == "sorted_grid_first_minimum"
    assert config.baselines.tie_tolerance == 1e-12
    for fitter in ("stationary", "ewma", "logistic"):
        assert np.array_equal(calls[fitter]["train_mask"], np.array([[False, True]]))
        assert np.array_equal(
            calls[fitter]["validation_mask"], np.array([[False, True]])
        )
    assert calls["ewma"]["args"][0].shape[-1] == calls["ewma"]["args"][1].shape[-1]
    assert calls["logistic"]["args"][0].shape[-1] == calls["logistic"]["args"][1].shape[-1]
    assert all(arm.temperature == 1.0 for arm in bundle.arms)
    assert bundle.arm_aliases == config.baselines.arm_aliases


def test_empirical_stationary_is_raw_mean_over_scored_train_only(
    monkeypatch: pytest.MonkeyPatch, partitions: DevelopmentPartitions
) -> None:
    config = load_identifiability_config(Path("configs/temporal_identifiability.json"))
    scored_train = (np.arange(2610) % 3 == 0).astype(np.uint8)

    def generate(identity: SequenceIdentity, _config: object):
        errors = np.zeros((2, 2610), dtype=np.uint8)
        if identity.role == "train":
            errors[0] = 1  # Burn-in must not enter the empirical field.
            errors[1] = scored_train
        elif identity.role == "validation":
            errors[:] = 1  # Validation must not alter the raw training mean.
        return _sequence_with_errors(identity, errors)

    monkeypatch.setattr("qldpc_fno.identifiability.baseline_bundle._generate", generate)
    monkeypatch.setattr(
        "qldpc_fno.identifiability.baseline_bundle.fit_ewma", lambda *a, **k: _ewma()
    )
    monkeypatch.setattr(
        "qldpc_fno.identifiability.baseline_bundle.fit_logistic_ar",
        lambda *a, **k: _logistic(),
    )

    stationary = fit_development_bundle(partitions, config).arm("empirical_stationary")

    assert stationary.shrinkage == 1.0
    assert stationary.scalar_probability == pytest.approx(float(scored_train.mean()))
    assert np.array_equal(stationary.empirical_field.reshape(-1), scored_train)


@pytest.mark.parametrize(
    ("score_improvement", "expected_decay", "expected_weight"),
    [(0.5e-12, 0.5, 1.0), (2.0e-12, 0.8, 2.0)],
)
def test_existing_fitters_apply_exact_first_minimum_tie_boundary(
    monkeypatch: pytest.MonkeyPatch,
    score_improvement: float,
    expected_decay: float,
    expected_weight: float,
) -> None:
    from qldpc_fno.temporal.baselines import fit_ewma

    calls = 0

    def tied_mapping(*args, **kwargs):
        nonlocal calls
        calls += 1
        return (
            np.full((1, 1, 1), calls),
            np.zeros(1),
            1e-4,
            1.0 - calls * score_improvement,
        )

    monkeypatch.setattr("qldpc_fno.temporal.baselines._select_mapping", tied_mapping)
    values = np.zeros((1, 1, 1, 1), dtype=np.float64)
    mask = np.ones((1, 1), dtype=np.bool_)

    model = fit_ewma(
        values,
        values,
        values,
        values,
        train_mask=mask,
        validation_mask=mask,
        decays=(0.8, 0.5),
        kernel_size=1,
        l2_grid=(1e-4,),
        max_iter=1,
    )

    assert model.decay == expected_decay
    assert np.array_equal(model.weight, np.full((1, 1, 1), expected_weight))


def test_baseline_geometry_losslessly_reshapes_raw_rings() -> None:
    raw = np.arange(945, dtype=np.uint8).reshape(1, 1, 1, 945) % 2
    assert np.array_equal(_syndrome_to_qubit_features(raw, 2610).reshape(-1), raw.reshape(-1))
    assert _syndrome_to_qubit_features(raw, 2610).shape == (1, 1, 21, 45)


def test_safe_bundle_round_trip_rejects_tampered_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, partitions: DevelopmentPartitions
) -> None:
    config = load_identifiability_config(Path("configs/temporal_identifiability.json"))
    monkeypatch.setattr("qldpc_fno.identifiability.baseline_bundle._generate", lambda identity, config: _sequence(identity))
    monkeypatch.setattr("qldpc_fno.identifiability.baseline_bundle.fit_stationary", lambda *a, **k: _stationary())
    monkeypatch.setattr("qldpc_fno.identifiability.baseline_bundle.fit_ewma", lambda *a, **k: _ewma())
    monkeypatch.setattr("qldpc_fno.identifiability.baseline_bundle.fit_logistic_ar", lambda *a, **k: _logistic())
    bundle = fit_development_bundle(partitions, config)

    manifest = write_frozen_bundle(tmp_path / "bundle", bundle)
    loaded = read_verified_bundle(tmp_path / "bundle", manifest)
    assert loaded.integrity_sha256 == bundle.integrity_sha256
    assert loaded.arm("ewma").weight.tobytes() == bundle.arm("ewma").weight.tobytes()

    metadata = tmp_path / "bundle" / "metadata.json"
    metadata.write_text(metadata.read_text().replace('"ewma"', '"renamed"', 1))
    with pytest.raises(ValueError, match="metadata|integrity|alias"):
        read_verified_bundle(tmp_path / "bundle", manifest)


def test_safe_bundle_is_deterministic_numeric_non_pickle_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, partitions: DevelopmentPartitions
) -> None:
    bundle = _bundle(monkeypatch, partitions)

    first = write_frozen_bundle(tmp_path / "first", bundle)
    second = write_frozen_bundle(tmp_path / "second", bundle)

    assert first == second
    assert (tmp_path / "first" / "metadata.json").read_bytes() == (
        tmp_path / "second" / "metadata.json"
    ).read_bytes()
    assert (tmp_path / "first" / "arrays.npz").read_bytes() == (
        tmp_path / "second" / "arrays.npz"
    ).read_bytes()
    with np.load(tmp_path / "first" / "arrays.npz", allow_pickle=False) as arrays:
        assert all(arrays[name].dtype != object for name in arrays.files)


def test_bundle_publication_failure_before_rename_leaves_no_final_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, partitions: DevelopmentPartitions
) -> None:
    bundle = _bundle(monkeypatch, partitions)
    destination = tmp_path / "bundle"

    def fail_before_rename(source: object, target: object) -> None:
        assert Path(target) == destination
        assert Path(source).is_dir()
        assert not destination.exists()
        raise OSError("injected pre-rename failure")

    monkeypatch.setattr("qldpc_fno.identifiability.baseline_bundle.os.replace", fail_before_rename)

    with pytest.raises(OSError, match="pre-rename"):
        write_frozen_bundle(destination, bundle)

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".bundle.*"))


def test_bundle_manifest_rejects_invalid_and_forged_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, partitions: DevelopmentPartitions
) -> None:
    with pytest.raises(ValueError, match="schema"):
        BundleManifest(2, "a" * 64, "b" * 64, "c" * 64, ())

    bundle = _bundle(monkeypatch, partitions)
    path = tmp_path / "bundle"
    valid = write_frozen_bundle(path, bundle)
    forged = BundleManifest.__new__(BundleManifest)
    for field in dataclasses.fields(valid):
        object.__setattr__(forged, field.name, getattr(valid, field.name))
    object.__setattr__(forged, "schema_version", 2)

    with pytest.raises(ValueError, match="schema"):
        read_verified_bundle(path, forged)


@pytest.mark.parametrize("missing", ["metadata.json", "arrays.npz"])
def test_safe_bundle_rejects_missing_payload_member(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    partitions: DevelopmentPartitions,
    missing: str,
) -> None:
    bundle = _bundle(monkeypatch, partitions)
    path = tmp_path / missing.replace(".", "_")
    manifest = write_frozen_bundle(path, bundle)
    (path / missing).unlink()

    with pytest.raises(ValueError, match="missing"):
        read_verified_bundle(path, manifest)


def test_safe_bundle_rejects_missing_or_renamed_safe_array(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, partitions: DevelopmentPartitions
) -> None:
    bundle = _bundle(monkeypatch, partitions)
    path = tmp_path / "bundle"
    manifest = write_frozen_bundle(path, bundle)
    arrays_path = path / "arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as loaded:
        arrays = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    arrays["renamed__weight"] = arrays.pop("ewma__weight")
    np.savez(arrays_path, **arrays)
    manifest = dataclasses.replace(
        manifest, arrays_sha256=hashlib.sha256(arrays_path.read_bytes()).hexdigest()
    )

    with pytest.raises(ValueError, match="missing|renamed"):
        read_verified_bundle(path, manifest)


@pytest.mark.parametrize("rehash_descriptor", [False, True])
def test_safe_bundle_rejects_tampered_array_even_when_outer_hashes_are_replaced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    partitions: DevelopmentPartitions,
    rehash_descriptor: bool,
) -> None:
    bundle = _bundle(monkeypatch, partitions)
    path = tmp_path / f"bundle-{rehash_descriptor}"
    manifest = write_frozen_bundle(path, bundle)
    arrays_path = path / "arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as loaded:
        arrays = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    arrays["ewma__weight"][0, 0, 0] += 1.0
    np.savez(arrays_path, **arrays)
    manifest = dataclasses.replace(
        manifest, arrays_sha256=hashlib.sha256(arrays_path.read_bytes()).hexdigest()
    )
    if rehash_descriptor:
        metadata_path = path / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        descriptor = metadata["arms"][1]["arrays"]["weight"]
        descriptor["sha256"] = _array_digest_for_test(arrays["ewma__weight"])
        metadata_path.write_text(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
        manifest = dataclasses.replace(
            manifest, metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        )

    with pytest.raises(ValueError, match="rehashed|tampered|integrity"):
        read_verified_bundle(path, manifest)


def _sequence(identity: SequenceIdentity):
    from qldpc_fno.identifiability.types import (
        ContemporaneousOracleInput,
        DeployableHistory,
        GeneratedSequence,
        LatentHistoryOracleInput,
        TrainingTargets,
    )

    return GeneratedSequence(
        identity=identity,
        deployable=DeployableHistory(np.zeros((2, 945), dtype=np.uint8), np.array([False, True])),
        latent_oracle=LatentHistoryOracleInput(np.array([0.0, 0.0])),
        contemporaneous_oracle=ContemporaneousOracleInput(np.full((2, 2610), 0.1)),
        targets=TrainingTargets(np.zeros((2, 2610), dtype=np.uint8), np.zeros((2, 1), dtype=np.uint8)),
    )


def _sequence_with_errors(identity: SequenceIdentity, errors: np.ndarray):
    from qldpc_fno.identifiability.types import (
        ContemporaneousOracleInput,
        DeployableHistory,
        GeneratedSequence,
        LatentHistoryOracleInput,
        TrainingTargets,
    )

    return GeneratedSequence(
        identity=identity,
        deployable=DeployableHistory(
            np.zeros((2, 945), dtype=np.uint8), np.array([False, True])
        ),
        latent_oracle=LatentHistoryOracleInput(np.array([0.0, 0.0])),
        contemporaneous_oracle=ContemporaneousOracleInput(np.full((2, 2610), 0.1)),
        targets=TrainingTargets(errors, np.zeros((2, 1), dtype=np.uint8)),
    )


def _bundle(
    monkeypatch: pytest.MonkeyPatch, partitions: DevelopmentPartitions
) -> FrozenEstimatorBundle:
    config = load_identifiability_config(Path("configs/temporal_identifiability.json"))
    monkeypatch.setattr(
        "qldpc_fno.identifiability.baseline_bundle._generate",
        lambda identity, config: _sequence(identity),
    )
    monkeypatch.setattr(
        "qldpc_fno.identifiability.baseline_bundle.fit_stationary",
        lambda *a, **k: _stationary(),
    )
    monkeypatch.setattr(
        "qldpc_fno.identifiability.baseline_bundle.fit_ewma", lambda *a, **k: _ewma()
    )
    monkeypatch.setattr(
        "qldpc_fno.identifiability.baseline_bundle.fit_logistic_ar",
        lambda *a, **k: _logistic(),
    )
    return fit_development_bundle(partitions, config)


def _array_digest_for_test(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _stationary():
    from qldpc_fno.temporal.baselines import StationaryForecaster

    return StationaryForecaster(0.1, np.array([[0.1]]), 1.0)


def _ewma():
    from qldpc_fno.temporal.baselines import CircularLogisticForecaster

    return CircularLogisticForecaster(np.zeros((1, 1, 5)), np.zeros(1), 1e-4, feature_kind="ewma", decay=0.5)


def _logistic():
    from qldpc_fno.temporal.baselines import CircularLogisticForecaster

    return CircularLogisticForecaster(np.zeros((1, 32, 3)), np.zeros(1), 1e-4, feature_kind="lagged", lags=32)
