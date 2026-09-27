from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from qldpc_fno.artifacts import sha256_file
from qldpc_fno.decision import simple_confirmation_io as io

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "configs/simple_planar_confirmation.json"
FIXTURE = ROOT / "configs/simple_planar_confirmation_fixture_shots.json"
SCIENTIFIC = ROOT / "configs/simple_planar_confirmation_shots.json"


def test_preflight_is_metadata_only(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("metadata preflight touched a scientific coordinate or computation")

    monkeypatch.setattr(io.planar_shot_data, "_shot_seed", forbidden)
    monkeypatch.setattr(io.planar_shot_data, "generate_planar_shots", forbidden)
    monkeypatch.setattr(io, "evaluate_shot", forbidden)
    record = io.preflight(ANALYSIS, SCIENTIFIC)
    assert record["contract_id"] == "simple_planar_confirmation_v1"
    assert record["unopened_domain"] == {
        "domain": "qldpc-fno/simple-planar-confirmation/v1",
        "campaign_seed": 2409828766515432030,
        "coordinates_derived": False,
    }
    assert record["expected_shots"] == 4096


def test_recursive_domain_inventory_includes_nested_keys(tmp_path: Path) -> None:
    config = tmp_path / "nested.json"
    config.write_text(json.dumps({"outer": [{"other_seed_domain": "nested/domain"}]}))
    inventory = io._historical_domain_inventory(tmp_path, excluded_paths=frozenset())
    assert inventory[config.name]["domains"] == ["nested/domain"]
    assert inventory[config.name]["sha256"] == sha256_file(config)


def test_recursive_domain_inventory_rejects_duplicate_domains(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(
        json.dumps({"seed_domain": "same", "nested": {"other_seed_domain": "same"}})
    )
    with pytest.raises(ValueError, match="duplicate historical seed domain"):
        io._historical_domain_inventory(tmp_path, excluded_paths=frozenset())


def test_historical_sampler_inventory_finds_nested_planar_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "old" / "planar_shots.json"
    artifact.parent.mkdir()
    artifact.write_text(json.dumps({"shots": [{"sampler_seed": 17}, {"sampler_seed": 23}]}))
    assert io._historical_sampler_seeds(tmp_path) == {17, 23}


def test_historical_sampler_inventory_rejects_duplicate_seed(tmp_path: Path) -> None:
    first = tmp_path / "one" / "planar_shots.json"
    second = tmp_path / "two" / "planar_shots.json"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text(json.dumps({"shots": [{"sampler_seed": 17}]}))
    second.write_text(json.dumps({"shots": [{"sampler_seed": 17}]}))
    with pytest.raises(ValueError, match="duplicate historical sampler seed"):
        io._historical_sampler_seeds(tmp_path)


def test_validate_release_rejects_changed_review_bytes(tmp_path: Path) -> None:
    review = tmp_path / "review.md"
    review.write_text("accepted\n")
    record = io.preflight(ANALYSIS, SCIENTIFIC)
    receipt = io._release_receipt_for_test(
        record, contract_review=review, implementation_review=review
    )
    path = tmp_path / "release.json"
    path.write_text(json.dumps(receipt))
    review.write_text("changed\n")
    with pytest.raises(ValueError, match="review binding"):
        io.validate_release(path, record)


def test_validate_release_rejects_foreign_preflight(tmp_path: Path) -> None:
    review = tmp_path / "review.md"
    review.write_text("accepted\n")
    record = io.preflight(ANALYSIS, SCIENTIFIC)
    receipt = io._release_receipt_for_test(
        record, contract_review=review, implementation_review=review
    )
    receipt["expected_shots"] = 2
    path = tmp_path / "release.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="release receipt"):
        io.validate_release(path, record)


def test_validate_release_rejects_duplicate_json_key(tmp_path: Path) -> None:
    path = tmp_path / "release.json"
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        io.validate_release(path, {})


def test_provenance_recheck_detects_changed_source(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = io._base_provenance(ANALYSIS, FIXTURE, release_binding=None)
    monkeypatch.setattr(io, "_source_hashes", lambda root: {"changed": "digest"})
    with pytest.raises(ValueError, match="provenance changed"):
        io._assert_base_provenance_current(baseline, ANALYSIS, FIXTURE)


def test_fixture_generate_publishes_exact_two_file_closure(tmp_path: Path) -> None:
    out = tmp_path / "generated"
    payload = io.generate(ANALYSIS, FIXTURE, out, release_path=None, fixture=True)
    assert {path.name for path in out.iterdir()} == {
        "planar_shots.json",
        "generation_manifest.json",
    }
    assert payload["manifest"]["shots_binding"]["sha256"] == sha256_file(out / "planar_shots.json")
    assert {"pyproject.toml", "uv.lock"}.issubset(payload["manifest"]["source_sha256"])


def test_fixture_cannot_escalate_to_scientific_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fixture"):
        io.generate(ANALYSIS, SCIENTIFIC, tmp_path / "out", release_path=None, fixture=True)


def test_generation_validation_rejects_manifest_extra_field(tmp_path: Path) -> None:
    out = tmp_path / "generated"
    io.generate(ANALYSIS, FIXTURE, out, release_path=None, fixture=True)
    manifest_path = out / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["unexpected"] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="generation manifest fields"):
        io._validate_generation_directory(out)


@pytest.mark.parametrize("kind", ["directory", "file", "symlink", "lock"])
def test_atomic_publication_refuses_preexisting_destination(tmp_path: Path, kind: str) -> None:
    out = tmp_path / "result"
    if kind == "directory":
        out.mkdir()
    elif kind == "file":
        out.write_text("old")
    elif kind == "symlink":
        out.symlink_to(tmp_path / "missing")
    else:
        (tmp_path / "result.lock").write_text("stale")
    with pytest.raises(FileExistsError):
        io._publish_directory(out, lambda stage: (stage / "value").write_text("new"))
    if kind == "file":
        assert out.read_text() == "old"


def test_atomic_publication_failure_leaves_receipt_not_partial(tmp_path: Path) -> None:
    out = tmp_path / "result"

    def fail(stage: Path) -> None:
        (stage / "partial").write_text("partial")
        raise RuntimeError("solver exploded")

    with pytest.raises(RuntimeError, match="solver exploded"):
        io._publish_directory(out, fail, approved_hashes={"input": "abc"})
    assert not out.exists()
    receipt = json.loads((tmp_path / "result.failure.json").read_text())
    assert receipt["exception_type"] == "RuntimeError"
    assert receipt["stage"] == "populate"
    assert receipt["approved_hashes"] == {"input": "abc"}
    assert not (tmp_path / "result.lock").exists()


def test_generate_failure_receipt_binds_sources_and_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "generated"

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("generator failed")

    monkeypatch.setattr(io.planar_shot_data, "generate_planar_shots", fail)
    with pytest.raises(RuntimeError, match="generator failed"):
        io.generate(ANALYSIS, FIXTURE, out, release_path=None, fixture=True)
    receipt = json.loads((tmp_path / "generated.failure.json").read_text())
    assert receipt["approved_hashes"]["config:analysis"] == sha256_file(ANALYSIS)
    assert receipt["approved_hashes"]["source:pyproject.toml"] == sha256_file(
        ROOT / "pyproject.toml"
    )


def test_atomic_publication_does_not_remove_foreign_replaced_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "result"
    lock = tmp_path / "result.lock"
    original_replace = os.replace

    def replace_lock(source: str | Path, destination: str | Path) -> None:
        lock.unlink()
        lock.write_text("foreign")
        original_replace(source, destination)

    monkeypatch.setattr(io.os, "replace", replace_lock)
    io._publish_directory(out, lambda stage: (stage / "value").write_text("ok"))
    assert lock.read_text() == "foreign"


def test_fixture_run_publishes_transactional_result(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    io.generate(ANALYSIS, FIXTURE, generated, release_path=None, fixture=True)
    out = tmp_path / "run"
    payload = io.run(
        ANALYSIS,
        FIXTURE,
        generated / "planar_shots.json",
        out,
        release_path=None,
        fixture=True,
    )
    assert payload["raw"]["status"] == "reduced_non_scientific"
    assert payload["raw"]["scientific_eligible"] is False
    assert {path.name for path in out.iterdir()} == {
        "result.json",
        "summary.json",
        "timing.json",
    }


def test_fixture_run_requires_generation_manifest(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    io.generate(ANALYSIS, FIXTURE, generated, release_path=None, fixture=True)
    (generated / "generation_manifest.json").unlink()
    with pytest.raises(ValueError, match="generation directory"):
        io.run(
            ANALYSIS,
            FIXTURE,
            generated / "planar_shots.json",
            tmp_path / "run",
            release_path=None,
            fixture=True,
        )
