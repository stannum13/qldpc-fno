from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.decoders.bplsd import DecodeBatchResult
from qldpc_fno.temporal import evaluation as evaluation_module
from qldpc_fno.temporal import screen as screen_module
from qldpc_fno.temporal.baselines import CircularLogisticForecaster, StationaryForecaster
from qldpc_fno.temporal.config import CausalExperimentConfig
from qldpc_fno.temporal.dataset import read_verified_sequence
from qldpc_fno.temporal.screen import (
    generate_sequence_campaign,
    run_reduced_screen,
    verify_screen_result,
    verify_sequence_campaign,
)
from qldpc_fno.training.causal_sequence import (
    CausalTrainingResult,
    ForecastMetrics,
    OverfitResult,
)

CONFIG_PATH = Path("configs/causal_fno_hippo_reduced.json")


def _tiny_config(path: Path) -> Path:
    payload = json.loads(CONFIG_PATH.read_text())
    payload["splits"] = {"train": 1, "validation": 1, "calibration": 1, "test": 0}
    payload["rounds"] = {"burn_in": 1, "scored": 1}
    payload["model"].update({"hidden_width": 4, "fno_modes": 2, "fir_history": 2, "hippo_order": 2})
    payload["optimizer"].update({"max_epochs": 1, "batch_size": 1})
    write_canonical_json(path, payload)
    return path


def _repository() -> screen_module._RepositoryEvidence:
    commit = subprocess.run(
        ["git", "-C", str(Path.cwd()), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return screen_module._RepositoryEvidence(Path.cwd(), commit)


def test_generate_and_directly_regenerate_all_a_to_e_role_artifacts(tmp_path: Path) -> None:
    config_path = _tiny_config(tmp_path / "config.json")
    output = tmp_path / "sequences"

    first = generate_sequence_campaign(
        config_path=config_path, output_dir=output, _repository=_repository()
    )
    verify_sequence_campaign(config_path=config_path, output_dir=output, regenerate=True)
    second = generate_sequence_campaign(
        config_path=config_path, output_dir=output, _repository=_repository()
    )

    assert first == second
    assert len(first["sequences"]) == 15
    assert (output / "manifest.json").is_file()
    assert list(dict.fromkeys(row["regime"] for row in first["sequences"])) == list(
        CausalExperimentConfig.from_json(config_path).regimes
    )
    assert {row["role"] for row in first["sequences"]} == {
        "train",
        "validation",
        "calibration",
    }

    changed = json.loads(config_path.read_text())
    changed["campaign_seed"] += 1
    write_canonical_json(config_path, changed)
    with pytest.raises(FileExistsError, match="completed differing"):
        generate_sequence_campaign(
            config_path=config_path,
            output_dir=output,
            _repository=_repository(),
        )


def test_reduced_screen_forbids_scoring_empty_history_round_zero(tmp_path: Path) -> None:
    config_path = _tiny_config(tmp_path / "config.json")
    payload = json.loads(config_path.read_text())
    payload["rounds"]["burn_in"] = 0
    write_canonical_json(config_path, payload)

    with pytest.raises(ValueError, match="round zero.*unscored"):
        generate_sequence_campaign(
            config_path=config_path,
            output_dir=tmp_path / "sequences",
            _repository=_repository(),
        )


def test_arbitrary_mapping_publication_is_not_a_public_trust_boundary() -> None:
    assert not hasattr(screen_module, "publish_screen_result")


def test_repository_identity_is_anchored_to_screen_module_and_requires_clean_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        del kwargs
        commands.append(command)
        stdout = "" if "status" in command else "a" * 40 + "\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout)

    monkeypatch.setattr(screen_module.subprocess, "run", fake_run)
    evidence = screen_module._repository_evidence()
    expected_root = Path(screen_module.__file__).resolve().parents[3]
    assert evidence.root == expected_root
    assert all(command[1:3] == ["-C", str(expected_root)] for command in commands)

    monkeypatch.setattr(
        screen_module.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout=" M tracked\n"),
    )
    with pytest.raises(RuntimeError, match="clean nonignored"):
        screen_module._repository_evidence()


def test_tiny_full_screen_replays_evidence_and_is_scientifically_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _tiny_config(tmp_path / "config.json")
    sequences = tmp_path / "sequences"
    repository = _repository()
    root = generate_sequence_campaign(
        config_path=config_path,
        output_dir=sequences,
        _repository=repository,
    )
    config = CausalExperimentConfig.from_json(config_path)
    code = screen_module._canonical_code()
    correction_by_syndrome: dict[bytes, np.ndarray] = {}
    for row in root["sequences"]:
        if row["role"] != "validation":
            continue
        observed, supervision, _, _ = read_verified_sequence(
            sequences / row["path"],
            config=config,
            code=code,
            expected_source_commit=repository.commit,
        )
        for syndrome, error in zip(observed.syndromes, supervision.errors, strict=True):
            correction_by_syndrome.setdefault(syndrome.reshape(-1).tobytes(), error.reshape(-1))

    def logistic(probability: float, *, feature_kind: str) -> CircularLogisticForecaster:
        channels = 21 if feature_kind == "ewma" else 21 * 32
        width = 5 if feature_kind == "ewma" else 3
        return CircularLogisticForecaster(
            weight=np.zeros((58, channels, width)),
            bias=np.full(58, np.log(probability / (1.0 - probability))),
            l2=1e-4,
            feature_kind=feature_kind,
            decay=0.9 if feature_kind == "ewma" else None,
            lags=32,
        )

    monkeypatch.setattr(
        evaluation_module,
        "fit_stationary",
        lambda *args, **kwargs: StationaryForecaster(0.1, np.full((58, 45), 0.1), 0.0),
    )
    monkeypatch.setattr(
        evaluation_module,
        "fit_ewma",
        lambda *args, **kwargs: logistic(0.1, feature_kind="ewma"),
    )
    monkeypatch.setattr(
        evaluation_module,
        "fit_logistic_ar",
        lambda *args, **kwargs: logistic(0.2, feature_kind="lagged"),
    )
    monkeypatch.setattr(evaluation_module, "calibrate_temperature", lambda *args: 1.0)
    monkeypatch.setattr(
        evaluation_module, "fit_calibration_temperature", lambda *args, **kwargs: 1.0
    )

    def fake_train(model, *, train, validation, config, partition):
        del train, validation, config
        with torch.no_grad():
            for value in model.state_dict().values():
                value.zero_()
        state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        return CausalTrainingResult(
            model_state_dict=state,
            best_epoch=0,
            best_validation_nll=0.1,
            training_nll_history=(0.1,),
            validation_nll_history=(0.1,),
            partition_digest=partition.digest,
            regime=partition.regime,
            train_content_digest=partition.train_content_digest,
            validation_content_digest=partition.validation_content_digest,
            calibration_content_digest=partition.calibration_content_digest,
        )

    monkeypatch.setattr(screen_module, "train_causal_forecaster", fake_train)
    monkeypatch.setattr(
        screen_module,
        "overfit_causal_forecaster",
        lambda *args, **kwargs: OverfitResult(1, ForecastMetrics(0.01, 1.0)),
    )

    def fake_decode(hx, syndromes, logical, *, error_channels, config):
        del error_channels
        assert config == evaluation_module.CANONICAL_BPLSD_CONFIG
        corrections = np.stack(
            [correction_by_syndrome[row.tobytes()] for row in np.asarray(syndromes)]
        ).astype(np.uint8)
        predicted = np.asarray(logical @ corrections.T).T % 2
        valid = np.all((np.asarray(hx @ corrections.T).T % 2) == syndromes, axis=1)
        count = corrections.shape[0]
        return DecodeBatchResult(
            corrections=corrections,
            predicted_observables=predicted,
            syndrome_valid=valid,
            converged=np.ones(count, dtype=np.bool_),
            iterations=np.ones(count, dtype=np.int64),
            setup_latency_seconds=np.full(count, 0.001),
            decode_latency_seconds=np.full(count, 0.002),
            latency_seconds=np.full(count, 0.003),
        )

    monkeypatch.setattr(evaluation_module, "decode_bplsd_prior_batch", fake_decode)
    first = tmp_path / "screen-first"
    second = tmp_path / "screen-second"
    run_reduced_screen(
        config_path=config_path,
        sequence_dir=sequences,
        output_dir=first,
        _repository=repository,
    )
    run_reduced_screen(
        config_path=config_path,
        sequence_dir=sequences,
        output_dir=second,
        _repository=repository,
    )
    assert (first / "results.json").read_bytes() == (second / "results.json").read_bytes()
    first_manifest = json.loads((first / "manifest.json").read_text())
    second_manifest = json.loads((second / "manifest.json").read_text())
    assert {
        name: digest for name, digest in first_manifest["payloads"].items() if name != "timing.json"
    } == {
        name: digest
        for name, digest in second_manifest["payloads"].items()
        if name != "timing.json"
    }
    assert (first / "timing.json").read_bytes() != (second / "timing.json").read_bytes()
    timing = json.loads((first / "timing.json").read_text())
    assert timing["scope"] == "engineering_measurement_no_speed_claim"
    arm_timing = timing["regimes"]["joint_in_basis"]["fno_hippo"]
    assert set(arm_timing) >= {
        "estimator_batch_seconds",
        "bp_lsd_per_round_p50_seconds",
        "end_to_end_estimated_per_round_p50_seconds",
    }

    predictor_dir = first / "evidence" / "joint_in_basis" / "fno_hippo" / "predictor"
    predictor_metadata_path = predictor_dir.parent / "predictor.json"
    original_predictor_payloads = {
        path: path.read_bytes() for path in (predictor_metadata_path, *predictor_dir.glob("*.npy"))
    }
    original_results = (first / "results.json").read_bytes()
    metadata = json.loads(predictor_metadata_path.read_text())
    arrays = {path.stem: np.load(path, allow_pickle=False) for path in predictor_dir.glob("*.npy")}
    first_state_key = next(
        row["key"]
        for row in metadata["state"]
        if np.issubdtype(arrays[row["key"]].dtype, np.floating)
    )
    arrays[first_state_key] = np.array(arrays[first_state_key], copy=True)
    arrays[first_state_key].flat[0] = -0.0
    restored = evaluation_module.restore_frozen_predictor(
        json.loads(predictor_metadata_path.read_text()),
        {path.stem: np.load(path, allow_pickle=False) for path in predictor_dir.glob("*.npy")},
    )
    substituted_config = replace(
        config,
        optimizer=replace(config.optimizer, training_seed=1801),
    )
    substituted = replace(
        restored,
        config=substituted_config,
        config_digest=evaluation_module._json_digest(substituted_config.to_dict()),
        artifact_digest="",
    )
    object.__setattr__(
        substituted,
        "artifact_digest",
        evaluation_module._frozen_arm_integrity(substituted),
    )
    substituted_metadata, substituted_arrays = evaluation_module.export_frozen_predictor(
        substituted
    )
    write_canonical_json(predictor_metadata_path, substituted_metadata)
    for name, values in substituted_arrays.items():
        with (predictor_dir / f"{name}.npy").open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
    substituted_results = json.loads(original_results)
    substituted_results["regimes"]["joint_in_basis"]["arms"]["fno_hippo"]["hashes"][
        "provenance"
    ] = evaluation_module._arm_provenance_digest(substituted)
    write_canonical_json(first / "results.json", substituted_results)
    substituted_manifest = json.loads(json.dumps(first_manifest))
    substituted_manifest["payloads"][str(predictor_metadata_path.relative_to(first))] = sha256_file(
        predictor_metadata_path
    )
    substituted_manifest["payloads"]["results.json"] = sha256_file(first / "results.json")
    write_canonical_json(first / "manifest.json", substituted_manifest)
    with pytest.raises(ValueError, match="locked experiment configuration"):
        verify_screen_result(
            first,
            config_path=config_path,
            sequence_dir=sequences,
            _repository=repository,
        )
    for path, payload in original_predictor_payloads.items():
        path.write_bytes(payload)
    (first / "results.json").write_bytes(original_results)
    write_canonical_json(first / "manifest.json", first_manifest)

    forged_state = tuple(
        evaluation_module._FrozenTensor.from_tensor(
            row["name"], torch.from_numpy(np.asarray(arrays[row["key"]]).copy())
        )
        for row in metadata["state"]
    )
    forged = replace(
        restored,
        checkpoint_sha256=evaluation_module._state_dict_sha256(
            {item.name: item.tensor() for item in forged_state}
        ),
        artifact_digest="",
        _state=forged_state,
    )
    object.__setattr__(forged, "artifact_digest", evaluation_module._frozen_arm_integrity(forged))
    assert forged.checkpoint_sha256 != restored.checkpoint_sha256
    _, _, _, attack_evaluation, _ = screen_module._load_regime_batches(
        config=config,
        sequence_dir=sequences,
        root=root,
        code=code,
        regime="joint_in_basis",
    )
    assert np.array_equal(
        restored.predict(
            attack_evaluation.syndromes,
            sequence_ids=attack_evaluation.sequence_ids,
        ),
        forged.predict(
            attack_evaluation.syndromes,
            sequence_ids=attack_evaluation.sequence_ids,
        ),
    )
    forged_metadata, forged_arrays = evaluation_module.export_frozen_predictor(forged)
    write_canonical_json(predictor_metadata_path, forged_metadata)
    for name, values in forged_arrays.items():
        with (predictor_dir / f"{name}.npy").open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
    attack_manifest = json.loads(json.dumps(first_manifest))
    for path in original_predictor_payloads:
        relative = str(path.relative_to(first))
        attack_manifest["payloads"][relative] = sha256_file(path)
    write_canonical_json(first / "manifest.json", attack_manifest)
    with pytest.raises(ValueError, match="retraining checkpoint.*byte-identical"):
        verify_screen_result(
            first,
            config_path=config_path,
            sequence_dir=sequences,
            _repository=repository,
        )
    for path, payload in original_predictor_payloads.items():
        path.write_bytes(payload)
    write_canonical_json(first / "manifest.json", first_manifest)

    second_result = json.loads((second / "results.json").read_text())
    second_result["regimes"]["joint_in_basis"]["arms"]["fno_hippo"]["forecast"]["overall_nll"] = (
        99.0
    )
    write_canonical_json(second / "results.json", second_result)
    second_manifest["payloads"]["results.json"] = sha256_file(second / "results.json")
    write_canonical_json(second / "manifest.json", second_manifest)
    with pytest.raises(ValueError, match="recomputation"):
        verify_screen_result(
            second,
            config_path=config_path,
            sequence_dir=sequences,
            _repository=repository,
        )

    trace = first / "evidence" / "joint_in_basis" / "fno_hippo" / "trace" / "corrections.npy"
    with trace.open("wb") as handle:
        np.save(handle, np.ones((1, 2610), dtype=np.uint8), allow_pickle=False)
    relative_trace = str(trace.relative_to(first))
    first_manifest["payloads"][relative_trace] = sha256_file(trace)
    write_canonical_json(first / "manifest.json", first_manifest)
    with pytest.raises(ValueError, match="exact replay"):
        verify_screen_result(
            first,
            config_path=config_path,
            sequence_dir=sequences,
            _repository=repository,
        )


@pytest.mark.parametrize(
    ("script", "command"),
    [
        ("experiments/19_generate_causal_sequences.py", "verify"),
        ("experiments/20_run_causal_factor_screen.py", "verify"),
    ],
)
def test_causal_cli_entrypoints_expose_verification_commands(script: str, command: str) -> None:
    result = subprocess.run(
        [sys.executable, script, command, "--help"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--out" in result.stdout
