from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from qldpc_fno.identifiability import sequence_store
from qldpc_fno.identifiability.seeds import identifiability_seed
from qldpc_fno.identifiability.types import (
    ContemporaneousOracleInput,
    DeployableHistory,
    GeneratedSequence,
    LatentHistoryOracleInput,
    SequenceIdentity,
    TrainingTargets,
)

CONFIG_PATH = Path("configs/temporal_identifiability.json")
CLI_PATH = Path("experiments/21_generate_temporal_identifiability.py")


def _cli_module():
    spec = importlib.util.spec_from_file_location("temporal_identifiability_cli", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FastCode:
    name = "test-code"
    ell = 1
    n = 3
    k = 1
    hx = sparse.csr_matrix((2, 3), dtype=np.uint8)
    hz = sparse.csr_matrix((2, 3), dtype=np.uint8)


def _fast_sequence(config, *, regime: str, role: str, sequence_index: int, code):
    del code
    rounds = 2
    latent_seed = identifiability_seed(
        config, regime=regime, role=role, sequence_index=sequence_index, stream="latent"
    )
    bernoulli_seed = identifiability_seed(
        config, regime=regime, role=role, sequence_index=sequence_index, stream="bernoulli"
    )
    arrays = {
        "global_log_odds": np.zeros(rounds),
        "probabilities": np.full((rounds, 3), 0.0375),
        "errors": np.zeros((rounds, 3), dtype=np.uint8),
        "syndromes": np.zeros((rounds, 2), dtype=np.uint8),
        "logical_flips": np.zeros((rounds, 1), dtype=np.uint8),
        "scored_mask": np.array([False, True]),
    }
    content = sequence_store._sequence_content_sha256(
        regime=regime,
        role=role,
        sequence_index=sequence_index,
        latent_seed=latent_seed,
        bernoulli_seed=bernoulli_seed,
        arrays=arrays,
    )
    return GeneratedSequence(
        identity=SequenceIdentity(
            regime=regime,
            role=role,
            sequence_index=sequence_index,
            latent_seed=latent_seed,
            bernoulli_seed=bernoulli_seed,
            content_sha256=content,
        ),
        deployable=DeployableHistory(
            syndromes=arrays["syndromes"],
            scored_mask=arrays["scored_mask"],
        ),
        latent_oracle=LatentHistoryOracleInput(global_log_odds=arrays["global_log_odds"]),
        contemporaneous_oracle=ContemporaneousOracleInput(
            probabilities=arrays["probabilities"]
        ),
        targets=TrainingTargets(
            errors=arrays["errors"],
            logical_flips=arrays["logical_flips"],
        ),
    )


def _passed_fisher(config, checks):
    del config, checks
    return {"status": "passed", "maximum_derivative_error": 0.0}


def _dependencies(commit: str = "a" * 40, *, fisher=_passed_fisher):
    return sequence_store.SequenceStoreDependencies(
        code_factory=_FastCode,
        sequence_factory=_fast_sequence,
        fisher_precheck=fisher,
        repository_evidence=lambda: sequence_store.RepositoryEvidence(Path.cwd(), commit),
        logical_x_factory=lambda hx, hz: sparse.csr_matrix((1, 3), dtype=np.uint8),
        require_canonical_code=False,
    )


def _approval(record: Path) -> Path:
    manifest = json.loads((record / "manifest.json").read_text())
    approval = {
        "schema_version": 1,
        "kind": "temporal_identifiability_manual_approval",
        "approved": True,
        "approver": "integration-test",
        "approved_at": "2026-09-05T00:00:00Z",
        "development_record_sha256": sequence_store.sha256_file(record / "manifest.json"),
        "development_identity_sha256": manifest["identity_sha256"],
    }
    path = record.parent / "approval.json"
    path.write_text(json.dumps(approval, sort_keys=True))
    return path


def _generate(module, out: Path, roles: str, *, dependencies, approval: Path | None = None, record: Path | None = None):
    argv = ["generate", "--config", str(CONFIG_PATH), "--out", str(out), "--roles", roles]
    if approval is not None:
        argv.extend(("--approval", str(approval)))
    if record is not None:
        argv.extend(("--development-record", str(record)))
    return module.run(argv, dependencies=dependencies)


def test_cli_generates_verifies_and_regenerates_role_separated_payloads(tmp_path: Path) -> None:
    module = _cli_module()
    output = tmp_path / "development"
    dependencies = _dependencies()

    first = _generate(module, output, "train,validation,calibration", dependencies=dependencies)
    initial_manifest = (output / "manifest.json").read_bytes()
    verified = module.run(
        ["verify", "--config", str(CONFIG_PATH), "--out", str(output), "--roles", "train,validation,calibration"],
        dependencies=dependencies,
    )
    second = _generate(module, output, "train,validation,calibration", dependencies=dependencies)

    assert first == verified == second
    assert (output / "manifest.json").read_bytes() == initial_manifest
    assert {row["role"] for row in first["sequences"]} == {"train", "validation", "calibration"}
    assert first["fisher_precheck"]["status"] == "passed"


def test_cli_rejects_incomplete_overwrite_corruption_and_manifest_tampering(tmp_path: Path) -> None:
    module = _cli_module()
    dependencies = _dependencies()
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(FileExistsError, match="incomplete"):
        _generate(module, incomplete, "train", dependencies=dependencies)

    output = tmp_path / "development"
    _generate(module, output, "train", dependencies=dependencies)
    with pytest.raises(FileExistsError, match="differing"):
        _generate(module, output, "train", dependencies=_dependencies("b" * 40))

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("code")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing|unknown"):
        module.run(
            ["verify", "--config", str(CONFIG_PATH), "--out", str(output), "--roles", "train"],
            dependencies=dependencies,
        )

    output = tmp_path / "corrupt"
    _generate(module, output, "train", dependencies=dependencies)
    payload = next(output.glob("train/*/*.npz"))
    payload.write_bytes(payload.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="SHA-256"):
        module.run(
            ["verify", "--config", str(CONFIG_PATH), "--out", str(output), "--roles", "train"],
            dependencies=dependencies,
        )


def test_complete_manifest_rejects_extra_fields_rehashed_rows_and_role_overlap(tmp_path: Path) -> None:
    module = _cli_module()
    dependencies = _dependencies()
    output = tmp_path / "development"
    _generate(module, output, "train,validation", dependencies=dependencies)
    manifest_path = output / "manifest.json"
    original = json.loads(manifest_path.read_text())

    for field in (
        "source",
        "config",
        "code",
        "retained_checks",
        "roles",
        "fisher_precheck",
        "sequences",
        "identity_sha256",
        "content_sha256",
    ):
        value = json.loads(json.dumps(original))
        value.pop(field)
        manifest_path.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="missing|unknown"):
            module.run(
                ["verify", "--config", str(CONFIG_PATH), "--out", str(output), "--roles", "train,validation"],
                dependencies=dependencies,
            )

    for mutate in (
        lambda value: value.update({"extra": True}),
        lambda value: value["retained_checks"].update({"matrix_sha256": "0" * 64}),
        lambda value: value["sequences"].append(dict(value["sequences"][0])),
    ):
        value = json.loads(json.dumps(original))
        mutate(value)
        manifest_path.write_text(json.dumps(value))
        with pytest.raises(ValueError):
            module.run(
                ["verify", "--config", str(CONFIG_PATH), "--out", str(output), "--roles", "train,validation"],
                dependencies=dependencies,
            )
    manifest_path.write_text(json.dumps(original))


def test_test_role_firewall_binds_approval_record_and_fisher_gate(tmp_path: Path) -> None:
    module = _cli_module()
    dependencies = _dependencies()
    development = tmp_path / "development"
    _generate(module, development, "train,validation,calibration", dependencies=dependencies)
    approval = _approval(development)

    with pytest.raises(ValueError, match="approval"):
        _generate(module, tmp_path / "no-approval", "test", dependencies=dependencies)
    with pytest.raises(ValueError, match="development"):
        _generate(module, tmp_path / "no-record", "test", dependencies=dependencies, approval=approval)

    test_output = tmp_path / "test"
    _generate(module, test_output, "test", dependencies=dependencies, approval=approval, record=development)
    root = json.loads((test_output / "manifest.json").read_text())
    assert root["approval"]["development_identity_sha256"] == json.loads(
        (development / "manifest.json").read_text()
    )["identity_sha256"]

    bad_approval = json.loads(approval.read_text())
    bad_approval["development_record_sha256"] = "0" * 64
    approval.write_text(json.dumps(bad_approval))
    with pytest.raises(ValueError, match="hash"):
        _generate(module, tmp_path / "bad-binding", "test", dependencies=dependencies, approval=approval, record=development)

    failed_development = tmp_path / "failed-development"
    _generate(
        module,
        failed_development,
        "train,validation,calibration",
        dependencies=_dependencies(fisher=lambda config, checks: {"status": "precheck_failed"}),
    )
    failed_approval = _approval(failed_development)
    with pytest.raises(ValueError, match="Fisher"):
        _generate(
            module,
            tmp_path / "failed-test",
            "test",
            dependencies=dependencies,
            approval=failed_approval,
            record=failed_development,
        )


def test_dirty_source_blocks_test_role_before_any_output_is_opened(tmp_path: Path) -> None:
    module = _cli_module()
    development = tmp_path / "development"
    dependencies = _dependencies()
    _generate(module, development, "train,validation,calibration", dependencies=dependencies)
    approval = _approval(development)
    dirty_dependencies = sequence_store.SequenceStoreDependencies(
        code_factory=_FastCode,
        sequence_factory=_fast_sequence,
        fisher_precheck=_passed_fisher,
        logical_x_factory=lambda hx, hz: sparse.csr_matrix((1, 3), dtype=np.uint8),
        repository_evidence=lambda: (_ for _ in ()).throw(
            RuntimeError("temporal identifiability publication requires a clean source tree")
        ),
        require_canonical_code=False,
    )
    output = tmp_path / "blocked-test"
    with pytest.raises(RuntimeError, match="clean source tree"):
        _generate(
            module,
            output,
            "test",
            dependencies=dirty_dependencies,
            approval=approval,
            record=development,
        )
    assert not output.exists()
