from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from qldpc_fno.decoders.bplsd import DecodeBatchResult
from qldpc_fno.identifiability import screen as screen_module
from qldpc_fno.identifiability import sequence_store
from qldpc_fno.identifiability.baseline_bundle import BundleManifest
from qldpc_fno.identifiability.config import load_identifiability_config
from qldpc_fno.identifiability.filters import ForecastResult
from qldpc_fno.identifiability.screen import ScreenDependencies, ScreenSourceEvidence

CLI_PATH = Path("experiments/22_run_temporal_identifiability.py")
CONFIG_PATH = Path("configs/temporal_identifiability.json")


def _cli_module():
    spec = importlib.util.spec_from_file_location("temporal_identifiability_run_cli", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_exposes_only_development_confirmation_and_verify() -> None:
    module = _cli_module()
    parser = module._parser()

    actions = next(action for action in parser._actions if action.dest == "command")
    assert set(actions.choices) == {"development", "confirmation", "verify"}


def test_screen_public_api_is_importable() -> None:
    from qldpc_fno.identifiability.screen import (
        ScreenDependencies,
        run_confirmation,
        run_development,
        verify_identifiability_run,
    )

    assert ScreenDependencies is not None
    assert callable(run_development)
    assert callable(run_confirmation)
    assert callable(verify_identifiability_run)


def test_deranged_history_plan_is_payload_free_and_never_self_pairs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _write_sequences(tmp_path / "sequences", ("test",))

    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("derangement planning loaded a sequence payload")

    monkeypatch.setattr(screen_module, "_load_sequence", forbidden_load)
    planned = screen_module._history_source_rows(manifest["sequences"], seed=13987031144127066471)

    assert len(planned) == 128
    for (role, regime, target_index), by_arm in planned.items():
        assert role == "test"
        assert set(by_arm) == {
            "grid_bayes",
            "ewma",
            "logistic_ar32",
            "parity_moment_ar",
        }
        assert all(source["role"] == role for source in by_arm.values())
        assert all(source["regime"] == regime for source in by_arm.values())
        assert all(source["sequence_index"] != target_index for source in by_arm.values())


class _FastCode:
    name = "fast"
    ell = 1
    n = 3
    k = 1
    hx = sparse.csr_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8))
    hz = sparse.csr_matrix((1, 3), dtype=np.uint8)


@dataclass(frozen=True)
class _FakeBundle:
    integrity_sha256: str = "b" * 64
    config_payload: dict[str, object] | None = None
    partitions: object = None


def _fisher_record() -> dict[str, object]:
    config = load_identifiability_config(CONFIG_PATH)
    return {
        "status": "passed",
        "provenance": {
            "domain": config.seeds.fisher_domain,
            "seed": config.seeds.fisher,
            "law": config.fisher.draw_law,
            "draws": config.fisher.draws,
        },
        "minimum_information": 1.0,
        "median_information": 2.0,
        "maximum_information": 3.0,
        "cramer_rao_minimum": 1.0,
        "cramer_rao_median": 0.5,
        "cramer_rao_maximum": 1.0 / 3.0,
        "maximum_derivative_error": 0.0,
        "failure_reasons": [],
    }


def _write_sequences(root: Path, roles: tuple[str, ...]) -> dict[str, object]:
    records: list[dict[str, object]] = []
    counts = {"train": 8, "validation": 8, "calibration": 8, "test": 64}
    for role in roles:
        for regime in ("stationary_iid", "temporal_uniform"):
            for index in range(counts[role]):
                relative = Path(role) / regime / f"sequence-{index:05d}.npz"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                state = np.array([0.0, 0.1 if regime == "temporal_uniform" else 0.0])
                probabilities = np.full((2, 3), 0.0375)
                probabilities[1] += state[1] * 0.01
                errors = np.zeros((2, 3), dtype=np.uint8)
                errors[1, index % 3] = 1
                syndromes = np.asarray(errors @ _FastCode.hx.T, dtype=np.uint8) % 2
                logical_flips = (
                    np.asarray(
                        errors @ np.array([[1, 0, 1]], dtype=np.uint8).T,
                        dtype=np.uint8,
                    )
                    % 2
                )
                scored_mask = np.array([False, True])
                np.savez(
                    path,
                    global_log_odds=state,
                    probabilities=probabilities,
                    errors=errors,
                    syndromes=syndromes,
                    logical_flips=logical_flips,
                    scored_mask=scored_mask,
                )
                content = hashlib.sha256(str(relative).encode()).hexdigest()
                records.append(
                    {
                        "regime": regime,
                        "role": role,
                        "sequence_index": index,
                        "path": str(relative),
                        "seeds": {
                            "latent": 3 * index + 1,
                            "bernoulli": 3 * index + 2,
                            "filter": 3 * index + 3,
                        },
                        "sequence_content_sha256": content,
                    }
                )
    manifest = {
        "source": {"commit": "a" * 40, "clean": True},
        "config": {"sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()},
        "code": {"name": "fast"},
        "retained_checks": {
            "algorithm_version": "greedy_disjoint_rows/v1",
            "matrix_sha256": "c" * 64,
            "row_indices": [0],
            "supports": [[0, 1]],
            "weights": [2],
            "covered_qubits": [0, 1],
            "content_sha256": "d" * 64,
        },
        "roles": list(roles),
        "fisher_precheck": _fisher_record(),
        "sequences": records,
        "identity_sha256": "e" * 64,
        "content_sha256": "f" * 64,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def _forecast_kernel(*, sequence, include_grid_diagnostic, **_kwargs):
    rounds = sequence.deployable.syndromes.shape[0]
    values: dict[str, ForecastResult] = {}
    normal = (
        "known_marginal",
        "empirical_stationary",
        "ewma",
        "logistic_ar32",
        "parity_moment_ar",
        "grid_bayes",
        "latent_history_oracle",
        "contemporaneous_oracle",
    )
    for offset, arm in enumerate(normal):
        values[arm] = ForecastResult(
            arm,
            np.full(rounds, 0.03 + offset * 0.001),
            None if arm == "contemporaneous_oracle" else np.zeros(rounds),
            4096 if arm == "known_marginal" else None,
        )
    if include_grid_diagnostic:
        values["grid_bayes_doubled"] = ForecastResult(
            "grid_bayes_doubled", np.full(rounds, 0.03501), np.zeros(rounds), 4096
        )
    for arm in ("grid_bayes", "ewma", "logistic_ar32", "parity_moment_ar"):
        key = f"{arm}__history_deranged"
        values[key] = ForecastResult(key, np.full(rounds, 0.04), np.zeros(rounds), None)
    return values


def _score_kernel(*, sequence, forecast, **_kwargs):
    del sequence
    offset = float(np.mean(forecast.probabilities))
    return {
        "expected_ce": offset,
        "latent_nmse": offset + 1.0,
        "retained_syndrome_nll": offset + 2.0,
        "calibration": {
            "counts": [3] + [0] * 9,
            "predicted_sums": [3 * offset] + [0.0] * 9,
            "latent_sums": [3 * 0.0375] + [0.0] * 9,
            "absolute_error": abs(offset - 0.0375),
            "bin_edges": np.linspace(1e-5, 0.25, 11).tolist(),
        },
    }


def _dependencies(events: list[str], cpu_values: list[float] | None = None) -> ScreenDependencies:
    clock = iter(cpu_values or [0.0] * 100_000)
    fitted: list[_FakeBundle] = []

    def verify_sequences(**kwargs):
        events.append(f"open:{kwargs['roles'][0]}")
        return json.loads((kwargs["output_dir"] / "manifest.json").read_text())

    def fit_bundle(partitions, config):
        events.append("fit")
        bundle = _FakeBundle(config_payload=asdict(config), partitions=partitions)
        fitted.append(bundle)
        return bundle

    def write_bundle(path, bundle):
        del bundle
        path.mkdir(parents=True)
        (path / "metadata.json").write_text("{}")
        np.savez(path / "arrays.npz", value=np.array([1.0]))
        events.append("bundle-written")
        return BundleManifest(
            1,
            hashlib.sha256((path / "metadata.json").read_bytes()).hexdigest(),
            hashlib.sha256((path / "arrays.npz").read_bytes()).hexdigest(),
            "b" * 64,
            ("value",),
        )

    def read_bundle(path, manifest):
        del path, manifest
        events.append("bundle-verified")
        return fitted[-1]

    return ScreenDependencies(
        source_evidence=lambda: ScreenSourceEvidence(Path.cwd(), "a" * 40, "9" * 64),
        verify_sequences=verify_sequences,
        code_factory=_FastCode,
        logical_x_factory=lambda hx, hz: sparse.csr_matrix([[1, 0, 1]], dtype=np.uint8),
        require_canonical_code=False,
        fit_bundle=fit_bundle,
        write_bundle=write_bundle,
        read_bundle=read_bundle,
        forecast_sequence=_forecast_kernel,
        score_sequence=_score_kernel,
        process_time=lambda: next(clock),
        wall_time=lambda: 10.0,
    )


def _approval(development: Path) -> Path:
    manifest = json.loads((development / "manifest.json").read_text())
    bundle = manifest["bundle"]
    value = {
        "schema_version": 1,
        "kind": "temporal_identifiability_manual_approval",
        "decision": "APPROVE",
        "reviewer": "integration-test",
        "reviewed_at": "2026-09-05T00:00:00Z",
        "source_commit": manifest["source"]["commit"],
        "source_tree_sha256": manifest["source"]["tree_sha256"],
        "config_sha256": manifest["config"]["path_sha256"],
        "code_sha256": hashlib.sha256(
            json.dumps(manifest["code"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "retained_checks_sha256": manifest["retained_checks"]["content_sha256"],
        "development_record_sha256": hashlib.sha256(
            (development / "manifest.json").read_bytes()
        ).hexdigest(),
        "development_identity_sha256": manifest["identity_sha256"],
        "bundle_integrity_sha256": bundle["integrity_sha256"],
        "bundle_metadata_sha256": bundle["metadata_sha256"],
        "bundle_arrays_sha256": bundle["arrays_sha256"],
    }
    path = development.parent / "APPROVAL.json"
    path.write_text(json.dumps(value, sort_keys=True))
    return path


def test_development_publishes_frozen_bundle_and_streamed_full_evidence(tmp_path: Path) -> None:
    module = _cli_module()
    sequences = tmp_path / "sequences"
    _write_sequences(sequences, ("train", "validation", "calibration"))
    events: list[str] = []
    output = tmp_path / "run"

    result = module.run(
        [
            "development",
            "--config",
            str(CONFIG_PATH),
            "--sequences",
            str(sequences),
            "--out",
            str(output),
        ],
        dependencies=_dependencies(events),
    )

    assert events[:2] == ["open:train", "fit"]
    assert result["mode"] == "development"
    assert result["status"] == "complete"
    assert result["claim"] == "engineering_measurement_no_speed_claim"
    assert result["decision"] is None
    assert result["bundle"]["integrity_sha256"] == "b" * 64
    assert result["source"]["tree_sha256"] == "9" * 64
    assert result["fisher_precheck"]["status"] == "passed"
    assert set(result["inference"]["grid_diagnostics"]) == {
        "validation/stationary_iid",
        "validation/temporal_uniform",
    }
    assert result["runtime"]["process_cpu_seconds"] >= 0.0
    assert result["runtime"]["wall_seconds"] >= 0.0
    assert result["runtime"]["peak_rss_bytes"] >= 0
    assert len(result["evidence"]) == 48
    first = result["evidence"][0]
    assert set(first["seeds"]) == {"latent", "bernoulli", "filter"}
    assert set(first["arms"]) == {
        "known_marginal",
        "empirical_stationary",
        "ewma",
        "logistic_ar32",
        "parity_moment_ar",
        "grid_bayes",
        "latent_history_oracle",
        "contemporaneous_oracle",
        "grid_bayes__history_deranged",
        "ewma__history_deranged",
        "logistic_ar32__history_deranged",
        "parity_moment_ar__history_deranged",
    }
    validation = next(row for row in result["evidence"] if row["role"] == "validation")
    assert set(validation["arms"]) == {*first["arms"], "grid_bayes_doubled"}
    for arm in first["arms"].values():
        assert set(arm) == {"forecast", "state", "interior_cells", "endpoints"}
        assert set(arm["endpoints"]) == {
            "expected_ce",
            "latent_nmse",
            "retained_syndrome_nll",
            "calibration",
        }
    assert (output / "manifest.json").is_file()
    assert (output / "estimator-bundle" / "metadata.json").is_file()
    assert len(list((output / "evidence").rglob("*.npz"))) == 48


def test_confirmation_validates_and_refits_before_opening_test_and_non_go_never_decodes(
    tmp_path: Path,
) -> None:
    module = _cli_module()
    development_sequences = tmp_path / "development-sequences"
    _write_sequences(development_sequences, ("train", "validation", "calibration"))
    events: list[str] = []
    dependencies = _dependencies(events)
    development = tmp_path / "development"
    module.run(
        [
            "development",
            "--config",
            str(CONFIG_PATH),
            "--sequences",
            str(development_sequences),
            "--out",
            str(development),
        ],
        dependencies=dependencies,
    )
    approval = _approval(development)
    test_sequences = tmp_path / "test-sequences"
    _write_sequences(test_sequences, ("test",))
    events.clear()

    def non_go_inference(**_kwargs):
        events.append("inference")
        return {
            "decision": {
                "invalid_controls": (),
                "limitation": None,
                "outcome": "INCONCLUSIVE-OBSERVER-GAP",
                "winning_arms": (),
            }
        }

    def forbidden_decoder(*_args, **_kwargs):
        raise AssertionError("non-GO confirmation constructed a decoder")

    dependencies = replace(
        dependencies,
        inference=non_go_inference,
        decoder=forbidden_decoder,
    )
    output = tmp_path / "confirmation"
    result = module.run(
        [
            "confirmation",
            "--config",
            str(CONFIG_PATH),
            "--sequences",
            str(test_sequences),
            "--out",
            str(output),
            "--development-record",
            str(development),
            "--approval",
            str(approval),
        ],
        dependencies=dependencies,
    )

    assert events.index("fit") < events.index("bundle-verified") < events.index("open:test")
    assert result["decision"]["outcome"] == "INCONCLUSIVE-OBSERVER-GAP"
    assert result["inference"]["metadata"] == {
        "sampling_unit": "independent_sequence",
        "bootstrap": "paired_centered_rademacher_wild_bootstrap_t",
        "bootstrap_draws": 10_000,
        "holm_family": ["grid_bayes", "ewma", "logistic_ar32", "parity_moment_ar"],
        "delta_nll": 0.00025,
    }
    assert result["decoder"] == {
        "status": "not_run_by_design",
        "arms": [],
        "comparisons": {},
    }
    assert all("grid_bayes_doubled" not in row["arms"] for row in result["evidence"])
    assert (
        result["development_record"]["identity_sha256"]
        == json.loads((development / "manifest.json").read_text())["identity_sha256"]
    )
    assert result["approval"]["decision"] == "APPROVE"


def test_confirmation_firewall_rejects_bad_approval_before_test_payload_open(
    tmp_path: Path,
) -> None:
    module = _cli_module()
    development_sequences = tmp_path / "development-sequences"
    _write_sequences(development_sequences, ("train", "validation", "calibration"))
    events: list[str] = []
    dependencies = _dependencies(events)
    development = tmp_path / "development"
    module.run(
        [
            "development",
            "--config",
            str(CONFIG_PATH),
            "--sequences",
            str(development_sequences),
            "--out",
            str(development),
        ],
        dependencies=dependencies,
    )
    approval = _approval(development)
    value = json.loads(approval.read_text())
    value["bundle_integrity_sha256"] = "0" * 64
    approval.write_text(json.dumps(value))
    test_sequences = tmp_path / "test-sequences"
    _write_sequences(test_sequences, ("test",))
    events.clear()

    with pytest.raises(ValueError, match="approval|bundle"):
        module.run(
            [
                "confirmation",
                "--config",
                str(CONFIG_PATH),
                "--sequences",
                str(test_sequences),
                "--out",
                str(tmp_path / "confirmation"),
                "--development-record",
                str(development),
                "--approval",
                str(approval),
            ],
            dependencies=dependencies,
        )
    assert not any(event == "open:test" for event in events)


def test_confirmation_firewall_rejects_foreign_bundle_state_before_test_open(
    tmp_path: Path,
) -> None:
    module = _cli_module()
    development_sequences = tmp_path / "development-sequences"
    _write_sequences(development_sequences, ("train", "validation", "calibration"))
    events: list[str] = []
    base = _dependencies(events)
    development = tmp_path / "development"
    module.run(
        [
            "development",
            "--config",
            str(CONFIG_PATH),
            "--sequences",
            str(development_sequences),
            "--out",
            str(development),
        ],
        dependencies=base,
    )
    approval = _approval(development)
    test_sequences = tmp_path / "test-sequences"
    _write_sequences(test_sequences, ("test",))
    events.clear()
    assert base.read_bundle is not None

    def foreign_bundle(path, manifest):
        loaded = base.read_bundle(path, manifest)
        return replace(loaded, config_payload={"foreign": True})

    with pytest.raises(ValueError, match="bundle|config|development"):
        module.run(
            [
                "confirmation",
                "--config",
                str(CONFIG_PATH),
                "--sequences",
                str(test_sequences),
                "--out",
                str(tmp_path / "confirmation"),
                "--development-record",
                str(development),
                "--approval",
                str(approval),
            ],
            dependencies=replace(base, read_bundle=foreign_bundle),
        )
    assert "open:test" not in events


def test_confirmation_requires_approved_validation_grid_convergence_before_test_open(
    tmp_path: Path,
) -> None:
    module = _cli_module()
    development_sequences = tmp_path / "development-sequences"
    _write_sequences(development_sequences, ("train", "validation", "calibration"))
    events: list[str] = []
    base = _dependencies(events)

    def nonconverged_forecasts(**kwargs):
        forecasts = dict(_forecast_kernel(**kwargs))
        if kwargs["include_grid_diagnostic"]:
            rounds = kwargs["sequence"].deployable.syndromes.shape[0]
            forecasts["grid_bayes_doubled"] = ForecastResult(
                "grid_bayes_doubled", np.full(rounds, 0.06), np.zeros(rounds), 4096
            )
        return forecasts

    development = tmp_path / "development"
    module.run(
        [
            "development",
            "--config",
            str(CONFIG_PATH),
            "--sequences",
            str(development_sequences),
            "--out",
            str(development),
        ],
        dependencies=replace(base, forecast_sequence=nonconverged_forecasts),
    )
    development_manifest = development / "manifest.json"
    falsified = json.loads(development_manifest.read_text())
    for regime in ("stationary_iid", "temporal_uniform"):
        diagnostic = falsified["inference"]["grid_diagnostics"][f"validation/{regime}"]
        diagnostic["doubled_mean_gain"] = diagnostic["nominal_mean_gain"]
        diagnostic["absolute_difference"] = 0.0
        diagnostic["passed"] = True
    falsified["completed_stages"]["inference_metadata"] = screen_module._digest(
        falsified["inference"]
    )
    falsified["identity_sha256"] = screen_module._digest(screen_module._identity_input(falsified))
    development_manifest.write_text(json.dumps(falsified, sort_keys=True))
    approval = _approval(development)
    test_sequences = tmp_path / "test-sequences"
    _write_sequences(test_sequences, ("test",))
    events.clear()

    with pytest.raises(ValueError, match="grid|convergence|validation"):
        module.run(
            [
                "confirmation",
                "--config",
                str(CONFIG_PATH),
                "--sequences",
                str(test_sequences),
                "--out",
                str(tmp_path / "confirmation"),
                "--development-record",
                str(development),
                "--approval",
                str(approval),
            ],
            dependencies=base,
        )
    assert "open:test" not in events


def test_sequence_publication_replays_the_approved_task8_development_contract(
    tmp_path: Path,
) -> None:
    module = _cli_module()
    sequences = tmp_path / "development-sequences"
    _write_sequences(sequences, ("train", "validation", "calibration"))
    development = tmp_path / "development"
    module.run(
        [
            "development",
            "--config",
            str(CONFIG_PATH),
            "--sequences",
            str(sequences),
            "--out",
            str(development),
        ],
        dependencies=_dependencies([]),
    )
    approval = _approval(development)
    development_root = json.loads((development / "manifest.json").read_text())
    expected_partitions = screen_module._partitions_from_run(development_root)
    expected_config = asdict(load_identifiability_config(CONFIG_PATH))

    dependencies = sequence_store.SequenceStoreDependencies(
        code_factory=_FastCode,
        sequence_factory=lambda *_args, **_kwargs: None,
        fisher_precheck=lambda *_args, **_kwargs: _fisher_record(),
        repository_evidence=lambda: sequence_store.RepositoryEvidence(Path.cwd(), "a" * 40),
        logical_x_factory=lambda _hx, _hz: sparse.csr_matrix([[1, 0, 1]], dtype=np.uint8),
        require_canonical_code=False,
        read_run_bundle=lambda _path, _manifest: _FakeBundle(
            config_payload=expected_config,
            partitions=expected_partitions,
        ),
    )
    binding = sequence_store._validate_development_record(
        development,
        approval,
        config_path=CONFIG_PATH,
        dependencies=dependencies,
        evidence=sequence_store.RepositoryEvidence(Path.cwd(), "a" * 40),
    )

    assert (
        binding["development_record_sha256"]
        == hashlib.sha256((development / "manifest.json").read_bytes()).hexdigest()
    )
    assert binding["approver"] == "integration-test"

    legacy_approval = tmp_path / "legacy-approval.json"
    legacy_approval.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "temporal_identifiability_manual_approval",
                "approved": True,
                "approver": "integration-test",
                "approved_at": "2026-09-05T00:00:00Z",
                "development_record_sha256": binding["development_record_sha256"],
                "development_identity_sha256": binding["development_identity_sha256"],
            }
        )
    )
    with pytest.raises(ValueError, match="missing|unknown|approval"):
        sequence_store._approval_binding(legacy_approval, development)

    malformed = json.loads((development / "manifest.json").read_text())
    malformed["evidence"][0]["arms"]["known_marginal"]["endpoints"]["calibration"]["bin_edges"] = [
        0.0
    ] * 11
    malformed["completed_stages"]["development_evidence"] = screen_module._digest(
        malformed["evidence"]
    )
    malformed["identity_sha256"] = screen_module._digest(screen_module._identity_input(malformed))
    malformed["content_sha256"] = screen_module._digest(screen_module._content_input(malformed))
    (development / "manifest.json").write_text(json.dumps(malformed, sort_keys=True))
    malformed_approval = _approval(development)
    with pytest.raises(ValueError, match="calibration|endpoint"):
        sequence_store._validate_development_record(
            development,
            malformed_approval,
            config_path=CONFIG_PATH,
            dependencies=dependencies,
            evidence=sequence_store.RepositoryEvidence(Path.cwd(), "a" * 40),
        )


def test_go_decodes_exact_winners_and_oracle_on_common_rounds_with_verified_labels(
    tmp_path: Path,
) -> None:
    module = _cli_module()
    development_sequences = tmp_path / "development-sequences"
    _write_sequences(development_sequences, ("train", "validation", "calibration"))
    events: list[str] = []
    base = _dependencies(events)
    development = tmp_path / "development"
    module.run(
        [
            "development",
            "--config",
            str(CONFIG_PATH),
            "--sequences",
            str(development_sequences),
            "--out",
            str(development),
        ],
        dependencies=base,
    )
    approval = _approval(development)
    test_sequences = tmp_path / "test-sequences"
    _write_sequences(test_sequences, ("test",))
    decoder_calls: list[tuple[tuple[int, ...], float]] = []

    def go_inference(**_kwargs):
        return {
            "decision": {
                "invalid_controls": (),
                "limitation": None,
                "outcome": "GO-TEMPORAL-IDENTIFIED",
                "winning_arms": ("ewma", "grid_bayes"),
            }
        }

    def decoder(hx, syndromes, logical_x, *, error_channels, config):
        del hx, logical_x, config
        syndrome_rows = np.asarray(syndromes, dtype=np.uint8)
        mapping = {(1, 0): (1, 0, 0), (1, 1): (0, 1, 0), (0, 1): (0, 0, 1)}
        corrections = np.asarray(
            [mapping[tuple(int(value) for value in row)] for row in syndrome_rows],
            dtype=np.uint8,
        )
        predicted = np.asarray(corrections @ np.array([[1, 0, 1]], dtype=np.uint8).T) % 2
        decoder_calls.append((tuple(range(len(syndrome_rows))), float(np.mean(error_channels))))
        shots = len(syndrome_rows)
        return DecodeBatchResult(
            corrections=corrections,
            predicted_observables=predicted,
            syndrome_valid=np.ones(shots, dtype=np.bool_),
            converged=np.ones(shots, dtype=np.bool_),
            iterations=np.ones(shots, dtype=np.int64),
            setup_latency_seconds=np.zeros(shots),
            decode_latency_seconds=np.zeros(shots),
            latency_seconds=np.zeros(shots),
        )

    dependencies = replace(base, inference=go_inference, decoder=decoder)
    output = tmp_path / "confirmation"
    result = module.run(
        [
            "confirmation",
            "--config",
            str(CONFIG_PATH),
            "--sequences",
            str(test_sequences),
            "--out",
            str(output),
            "--development-record",
            str(development),
            "--approval",
            str(approval),
        ],
        dependencies=dependencies,
    )

    expected_arms = [
        "known_marginal",
        "grid_bayes",
        "ewma",
        "contemporaneous_oracle",
    ]
    assert result["decoder"]["status"] == "completed"
    assert result["decoder"]["arms"] == expected_arms
    assert len(decoder_calls) == 64 * len(expected_arms)
    memberships = {tuple(row["membership"]) for row in result["decoder"]["outcomes"]}
    assert memberships == {(1,)}
    common_rows = [
        {
            "role": row["role"],
            "regime": row["regime"],
            "sequence_index": row["sequence_index"],
            "sequence_content_sha256": row["sequence_content_sha256"],
            "membership": row["membership"],
        }
        for row in result["decoder"]["outcomes"]
        if row["arm"] == "known_marginal"
    ]
    assert result["decoder"]["common_sequence_round_identity_sha256"] == (
        screen_module._digest(common_rows)
    )
    for row in result["decoder"]["outcomes"]:
        with np.load(output / row["path"], allow_pickle=False) as payload:
            assert payload["syndrome_valid"].tolist() == [True]
            assert payload["logical_failure"].tolist() == [False]
            assert payload["corrections"].shape == (1, 3)
    assert set(result["decoder"]["comparisons"]) == {
        "grid_bayes",
        "ewma",
        "contemporaneous_oracle",
    }

    undeclared = output / "decoder" / "undeclared.bin"
    undeclared.write_bytes(b"not bound by the completion manifest")
    with pytest.raises(ValueError, match="undeclared|publication"):
        module.run(
            [
                "verify",
                "--config",
                str(CONFIG_PATH),
                "--sequences",
                str(test_sequences),
                "--out",
                str(output),
                "--development-record",
                str(development),
                "--approval",
                str(approval),
            ],
            dependencies=dependencies,
        )
    undeclared.unlink()

    decoder_payload = next((output / "decoder").rglob("*.npz"))
    decoder_payload.write_bytes(decoder_payload.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="decoder|payload|SHA-256"):
        module.run(
            [
                "verify",
                "--config",
                str(CONFIG_PATH),
                "--sequences",
                str(test_sequences),
                "--out",
                str(output),
                "--development-record",
                str(development),
                "--approval",
                str(approval),
            ],
            dependencies=dependencies,
        )


def test_verify_independently_refits_recomputes_and_rejects_tampering(tmp_path: Path) -> None:
    module = _cli_module()
    sequences = tmp_path / "development-sequences"
    _write_sequences(sequences, ("train", "validation", "calibration"))
    events: list[str] = []
    dependencies = _dependencies(events)
    output = tmp_path / "development"
    published = module.run(
        [
            "development",
            "--config",
            str(CONFIG_PATH),
            "--sequences",
            str(sequences),
            "--out",
            str(output),
        ],
        dependencies=dependencies,
    )
    events.clear()

    verified = module.run(
        [
            "verify",
            "--config",
            str(CONFIG_PATH),
            "--sequences",
            str(sequences),
            "--out",
            str(output),
        ],
        dependencies=dependencies,
    )

    assert verified == published
    assert "fit" in events
    assert "bundle-verified" in events

    manifest_path = output / "manifest.json"
    original_manifest = manifest_path.read_bytes()
    manifest = json.loads(original_manifest)
    manifest["completed_stages"] = {}
    manifest["identity_sha256"] = screen_module._digest(screen_module._identity_input(manifest))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="stage"):
        module.run(
            [
                "verify",
                "--config",
                str(CONFIG_PATH),
                "--sequences",
                str(sequences),
                "--out",
                str(output),
            ],
            dependencies=dependencies,
        )
    manifest_path.write_bytes(original_manifest)

    manifest = json.loads(original_manifest)
    manifest["runtime"]["deadline_process_cpu_seconds"] = 0
    manifest["identity_sha256"] = screen_module._digest(screen_module._identity_input(manifest))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="runtime|deadline"):
        module.run(
            [
                "verify",
                "--config",
                str(CONFIG_PATH),
                "--sequences",
                str(sequences),
                "--out",
                str(output),
            ],
            dependencies=dependencies,
        )
    manifest_path.write_bytes(original_manifest)

    manifest = json.loads(original_manifest)
    manifest["runtime"]["process_cpu_seconds"] = 21_601.0
    manifest["identity_sha256"] = screen_module._digest(screen_module._identity_input(manifest))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="runtime|deadline"):
        module.run(
            [
                "verify",
                "--config",
                str(CONFIG_PATH),
                "--sequences",
                str(sequences),
                "--out",
                str(output),
            ],
            dependencies=dependencies,
        )
    manifest_path.write_bytes(original_manifest)

    payload = next((output / "evidence").rglob("*.npz"))
    payload.write_bytes(payload.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA-256|evidence|payload"):
        module.run(
            [
                "verify",
                "--config",
                str(CONFIG_PATH),
                "--sequences",
                str(sequences),
                "--out",
                str(output),
            ],
            dependencies=dependencies,
        )


def test_cpu_deadline_publishes_atomic_abort_without_verdict_and_cannot_resume(
    tmp_path: Path,
) -> None:
    module = _cli_module()
    sequences = tmp_path / "development-sequences"
    _write_sequences(sequences, ("train", "validation", "calibration"))
    output = tmp_path / "development"
    result = module.run(
        [
            "development",
            "--config",
            str(CONFIG_PATH),
            "--sequences",
            str(sequences),
            "--out",
            str(output),
        ],
        dependencies=_dependencies([], [0.0, 0.0, 21_601.0, 21_601.0]),
    )

    assert result["status"] == "aborted_no_verdict"
    assert result["complete"] is False
    assert result["decision"] is None
    assert result["completed_stages"]
    assert {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()} == {
        "manifest.json"
    }
    with pytest.raises(FileExistsError, match="overwrite|resume"):
        module.run(
            [
                "development",
                "--config",
                str(CONFIG_PATH),
                "--sequences",
                str(sequences),
                "--out",
                str(output),
            ],
            dependencies=_dependencies([]),
        )


def test_deadline_crossed_while_finalizing_cannot_publish_a_verdict(tmp_path: Path) -> None:
    module = _cli_module()
    sequences = tmp_path / "development-sequences"
    _write_sequences(sequences, ("train", "validation", "calibration"))
    state = {"wall_calls": 0, "expired": False}

    def process_time() -> float:
        return 21_601.0 if state["expired"] else 0.0

    def wall_time() -> float:
        state["wall_calls"] += 1
        if state["wall_calls"] == 2:
            state["expired"] = True
        return float(state["wall_calls"])

    dependencies = replace(_dependencies([]), process_time=process_time, wall_time=wall_time)
    output = tmp_path / "development"

    result = module.run(
        [
            "development",
            "--config",
            str(CONFIG_PATH),
            "--sequences",
            str(sequences),
            "--out",
            str(output),
        ],
        dependencies=dependencies,
    )

    assert result["status"] == "aborted_no_verdict"
    assert result["decision"] is None
    assert {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()} == {
        "manifest.json"
    }


def test_confirmation_verify_replays_development_before_test_and_result_access(
    tmp_path: Path,
) -> None:
    module = _cli_module()
    development_sequences = tmp_path / "development-sequences"
    _write_sequences(development_sequences, ("train", "validation", "calibration"))
    events: list[str] = []
    base = _dependencies(events)
    development = tmp_path / "development"
    module.run(
        [
            "development",
            "--config",
            str(CONFIG_PATH),
            "--sequences",
            str(development_sequences),
            "--out",
            str(development),
        ],
        dependencies=base,
    )
    approval = _approval(development)
    test_sequences = tmp_path / "test-sequences"
    _write_sequences(test_sequences, ("test",))

    def non_go(**_kwargs):
        return {
            "decision": {
                "invalid_controls": (),
                "limitation": None,
                "outcome": "STOP-NO-PRACTICAL-CAUSAL-HEADROOM",
                "winning_arms": (),
            }
        }

    dependencies = replace(base, inference=non_go)
    confirmation = tmp_path / "confirmation"
    module.run(
        [
            "confirmation",
            "--config",
            str(CONFIG_PATH),
            "--sequences",
            str(test_sequences),
            "--out",
            str(confirmation),
            "--development-record",
            str(development),
            "--approval",
            str(approval),
        ],
        dependencies=dependencies,
    )
    events.clear()

    verified = module.run(
        [
            "verify",
            "--config",
            str(CONFIG_PATH),
            "--sequences",
            str(test_sequences),
            "--out",
            str(confirmation),
            "--development-record",
            str(development),
            "--approval",
            str(approval),
        ],
        dependencies=dependencies,
    )

    assert events.index("fit") < events.index("bundle-verified") < events.index("open:test")
    assert verified["decision"]["outcome"] == "STOP-NO-PRACTICAL-CAUSAL-HEADROOM"
