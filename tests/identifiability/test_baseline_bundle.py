from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qldpc_fno.identifiability.baseline_bundle import (
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
    assert calls["logistic"]["lags"] == config.baselines.logistic_lags
    assert calls["logistic"]["kernel_size"] == config.baselines.logistic_kernel
    assert calls["logistic"]["l2_grid"] == config.baselines.logistic_l2
    assert calls["logistic"]["max_iter"] == config.baselines.lbfgs_max_iter
    assert calls["ewma"]["args"][0].shape[-1] == calls["ewma"]["args"][1].shape[-1]
    assert calls["logistic"]["args"][0].shape[-1] == calls["logistic"]["args"][1].shape[-1]
    assert all(arm.temperature == 1.0 for arm in bundle.arms)
    assert bundle.arm_aliases == config.baselines.arm_aliases


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


def _stationary():
    from qldpc_fno.temporal.baselines import StationaryForecaster

    return StationaryForecaster(0.1, np.array([[0.1]]), 1.0)


def _ewma():
    from qldpc_fno.temporal.baselines import CircularLogisticForecaster

    return CircularLogisticForecaster(np.zeros((1, 1, 5)), np.zeros(1), 1e-4, feature_kind="ewma", decay=0.5)


def _logistic():
    from qldpc_fno.temporal.baselines import CircularLogisticForecaster

    return CircularLogisticForecaster(np.zeros((1, 32, 3)), np.zeros(1), 1e-4, feature_kind="lagged", lags=32)
