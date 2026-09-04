import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from qldpc_fno.codes.lifted_product import CSSCode, build_self_lifted_product
from qldpc_fno.codes.seeds import PAPER_LP_3_7_16
from qldpc_fno.temporal.config import CausalExperimentConfig
from qldpc_fno.temporal.dataset import (
    build_sequence_manifest,
    read_verified_sequence,
    regenerate_and_verify,
    write_sequence,
)
from qldpc_fno.temporal.generator import generate_latent_sequence, sample_sequence

CONFIG_PATH = Path("configs/causal_fno_hippo_reduced.json")


def _rewrite_payload_with_current_hash(
    artifact: Path, payload_name: str, mutate
) -> None:
    payload_path = artifact / payload_name
    with np.load(payload_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    mutate(arrays)
    with payload_path.open("wb") as handle:
        np.savez(handle, **arrays)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["payloads"][payload_name]["sha256"] = hashlib.sha256(
        payload_path.read_bytes()
    ).hexdigest()
    manifest["payloads"][payload_name]["arrays"] = {
        name: {"shape": list(array.shape), "dtype": array.dtype.str}
        for name, array in arrays.items()
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


@pytest.fixture
def config() -> CausalExperimentConfig:
    return CausalExperimentConfig.from_json(CONFIG_PATH)


@pytest.fixture(scope="module")
def code() -> CSSCode:
    return build_self_lifted_product(PAPER_LP_3_7_16)


def _materialize(config: CausalExperimentConfig, code: CSSCode):
    latent = generate_latent_sequence(
        config, regime="joint_basis_mismatch", role="validation", sequence_index=1
    )
    observed, supervision, diagnostics = sample_sequence(
        latent, bernoulli_seed=latent.seeds.bernoulli, code=code
    )
    manifest = build_sequence_manifest(config=config, latent=latent, code=code)
    return latent, observed, supervision, diagnostics, manifest


def test_round_trip_binds_payload_shapes_dtypes_hashes_and_identity(
    tmp_path: Path, config: CausalExperimentConfig, code: CSSCode
) -> None:
    _, observed, supervision, diagnostics, manifest = _materialize(config, code)
    artifact = tmp_path / "sequence"
    write_sequence(artifact, observed, supervision, diagnostics, manifest)

    loaded_observed, loaded_supervision, loaded_diagnostics, loaded_manifest = (
        read_verified_sequence(artifact)
    )

    assert np.array_equal(loaded_observed.syndromes, observed.syndromes)
    assert np.array_equal(loaded_supervision.errors, supervision.errors)
    assert np.array_equal(loaded_diagnostics.probabilities, diagnostics.probabilities)
    assert loaded_manifest["identity"] == {
        "regime": "joint_basis_mismatch",
        "role": "validation",
        "sequence_index": 1,
    }
    assert loaded_manifest["seeds"] == {
        "latent": manifest["seeds"]["latent"],
        "bernoulli": manifest["seeds"]["bernoulli"],
    }
    assert set(loaded_manifest["payloads"]) == {
        "diagnostics.npz",
        "observed.npz",
        "supervision.npz",
    }
    for payload in loaded_manifest["payloads"].values():
        assert len(payload["sha256"]) == 64
        assert payload["arrays"]
        for metadata in payload["arrays"].values():
            assert set(metadata) == {"dtype", "shape"}


def test_manifest_rejects_spoofed_lp_matrix_identity(
    config: CausalExperimentConfig, code: CSSCode
) -> None:
    latent = generate_latent_sequence(
        config, regime="stationary_iid", role="train", sequence_index=0
    )
    spoofed = replace(code, hz=sparse.csr_matrix(code.hz.shape, dtype=np.uint8))

    with pytest.raises(ValueError, match="matrix identity"):
        build_sequence_manifest(config=config, latent=latent, code=spoofed)


def test_publication_is_payload_first_manifest_last_and_refuses_overwrite(
    tmp_path: Path, config: CausalExperimentConfig, code: CSSCode
) -> None:
    _, observed, supervision, diagnostics, manifest = _materialize(config, code)
    artifact = tmp_path / "sequence"
    write_sequence(artifact, observed, supervision, diagnostics, manifest)

    assert (artifact / "manifest.json").exists()
    with pytest.raises(FileExistsError, match="refuse to overwrite"):
        write_sequence(artifact, observed, supervision, diagnostics, manifest)


def test_writer_rejects_malformed_identity_manifest_before_publication(
    tmp_path: Path, config: CausalExperimentConfig, code: CSSCode
) -> None:
    _, observed, supervision, diagnostics, manifest = _materialize(config, code)
    artifact = tmp_path / "sequence"
    malformed = {key: value for key, value in manifest.items() if key != "seeds"}

    with pytest.raises(ValueError, match="uncompleted sequence manifest"):
        write_sequence(artifact, observed, supervision, diagnostics, malformed)
    assert not artifact.exists()


def test_reader_rejects_incomplete_payload_only_directory(tmp_path: Path) -> None:
    artifact = tmp_path / "interrupted"
    artifact.mkdir()
    np.savez(artifact / "observed.npz", syndromes=np.zeros((1, 1, 1)))

    with pytest.raises(ValueError, match="incomplete sequence artifact"):
        read_verified_sequence(artifact)


def test_reader_rejects_one_byte_payload_corruption(
    tmp_path: Path, config: CausalExperimentConfig, code: CSSCode
) -> None:
    _, observed, supervision, diagnostics, manifest = _materialize(config, code)
    artifact = tmp_path / "sequence"
    write_sequence(artifact, observed, supervision, diagnostics, manifest)
    payload = artifact / "supervision.npz"
    content = bytearray(payload.read_bytes())
    content[len(content) // 2] ^= 1
    payload.write_bytes(content)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_verified_sequence(artifact)


def test_reader_rejects_manifest_shape_mismatch(
    tmp_path: Path, config: CausalExperimentConfig, code: CSSCode
) -> None:
    _, observed, supervision, diagnostics, manifest = _materialize(config, code)
    artifact = tmp_path / "sequence"
    write_sequence(artifact, observed, supervision, diagnostics, manifest)
    manifest_path = artifact / "manifest.json"
    contents = json.loads(manifest_path.read_text())
    contents["payloads"]["observed.npz"]["arrays"]["syndromes"]["shape"] = [1, 2, 3]
    manifest_path.write_text(json.dumps(contents))

    with pytest.raises(ValueError, match="shape mismatch"):
        read_verified_sequence(artifact)


def test_writer_rejects_cross_payload_round_mismatch_before_publication(
    tmp_path: Path, config: CausalExperimentConfig, code: CSSCode
) -> None:
    _, observed, supervision, diagnostics, manifest = _materialize(config, code)
    bad_supervision = replace(supervision, errors=supervision.errors[:-1])
    artifact = tmp_path / "sequence"

    with pytest.raises(ValueError, match="round dimensions must agree"):
        write_sequence(artifact, observed, bad_supervision, diagnostics, manifest)
    assert not artifact.exists()


def test_writer_rejects_payload_geometry_inconsistent_with_manifest_code(
    tmp_path: Path, config: CausalExperimentConfig, code: CSSCode
) -> None:
    _, observed, supervision, diagnostics, manifest = _materialize(config, code)
    malformed_supervision = replace(
        supervision,
        errors=supervision.errors[:, :1, :],
        logical_flips=supervision.logical_flips[:, :1],
    )
    malformed_diagnostics = replace(
        diagnostics,
        probabilities=diagnostics.probabilities[:, :1, :],
        channel_offsets=diagnostics.channel_offsets[:1],
    )
    artifact = tmp_path / "sequence"

    with pytest.raises(ValueError, match="canonical physical-error geometry"):
        write_sequence(
            artifact,
            observed,
            malformed_supervision,
            malformed_diagnostics,
            manifest,
        )
    assert not artifact.exists()


@pytest.mark.parametrize("field,value", [("ell", 45.0), ("n", True), ("k", 743)])
def test_writer_strictly_validates_manifest_code_types_and_geometry(
    tmp_path: Path,
    config: CausalExperimentConfig,
    code: CSSCode,
    field: str,
    value: object,
) -> None:
    _, observed, supervision, diagnostics, manifest = _materialize(config, code)
    manifest["code"][field] = value
    artifact = tmp_path / "sequence"

    with pytest.raises(ValueError, match="canonical code identity"):
        write_sequence(artifact, observed, supervision, diagnostics, manifest)
    assert not artifact.exists()


@pytest.mark.parametrize(
    ("payload_name", "mutate", "match"),
    [
        (
            "observed.npz",
            lambda arrays: arrays["syndromes"].__setitem__((0, 0, 0), 2),
            "syndromes must be binary",
        ),
        (
            "supervision.npz",
            lambda arrays: arrays["logical_flips"].__setitem__(
                (0, 0), 1 - arrays["logical_flips"][0, 0]
            ),
            "logical flips do not match",
        ),
        (
            "diagnostics.npz",
            lambda arrays: arrays["probabilities"].__setitem__((0, 0, 0), np.nan),
            "probabilities must be finite",
        ),
        (
            "diagnostics.npz",
            lambda arrays: arrays["event_onset"].__setitem__(0, True),
            "event onset must imply an active age-zero event",
        ),
    ],
)
def test_reader_rejects_semantic_tampering_even_when_hash_and_metadata_are_updated(
    tmp_path: Path,
    config: CausalExperimentConfig,
    code: CSSCode,
    payload_name: str,
    mutate,
    match: str,
) -> None:
    _, observed, supervision, diagnostics, manifest = _materialize(config, code)
    artifact = tmp_path / "sequence"
    write_sequence(artifact, observed, supervision, diagnostics, manifest)
    _rewrite_payload_with_current_hash(artifact, payload_name, mutate)

    with pytest.raises(ValueError, match=match):
        read_verified_sequence(artifact)


def test_reader_rejects_self_consistent_small_payloads_against_canonical_manifest(
    tmp_path: Path, config: CausalExperimentConfig, code: CSSCode
) -> None:
    _, observed, supervision, diagnostics, manifest = _materialize(config, code)
    artifact = tmp_path / "sequence"
    write_sequence(artifact, observed, supervision, diagnostics, manifest)
    _rewrite_payload_with_current_hash(
        artifact,
        "supervision.npz",
        lambda arrays: arrays.update(
            errors=arrays["errors"][:, :1, :],
            logical_flips=arrays["logical_flips"][:, :1],
        ),
    )
    _rewrite_payload_with_current_hash(
        artifact,
        "diagnostics.npz",
        lambda arrays: arrays.update(
            probabilities=arrays["probabilities"][:, :1, :],
            channel_offsets=arrays["channel_offsets"][:1],
        ),
    )

    with pytest.raises(ValueError, match="canonical physical-error geometry"):
        read_verified_sequence(artifact)


def test_reader_rejects_tampered_manifest_code_type_before_payload_acceptance(
    tmp_path: Path, config: CausalExperimentConfig, code: CSSCode
) -> None:
    _, observed, supervision, diagnostics, manifest = _materialize(config, code)
    artifact = tmp_path / "sequence"
    write_sequence(artifact, observed, supervision, diagnostics, manifest)
    manifest_path = artifact / "manifest.json"
    contents = json.loads(manifest_path.read_text())
    contents["code"]["ell"] = 45.0
    manifest_path.write_text(json.dumps(contents, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="canonical code identity"):
        read_verified_sequence(artifact)


def test_reader_rejects_unexpected_files_in_completed_artifact(
    tmp_path: Path, config: CausalExperimentConfig, code: CSSCode
) -> None:
    _, observed, supervision, diagnostics, manifest = _materialize(config, code)
    artifact = tmp_path / "sequence"
    write_sequence(artifact, observed, supervision, diagnostics, manifest)
    (artifact / "undeclared.bin").write_bytes(b"surprise")

    with pytest.raises(ValueError, match="undeclared files"):
        read_verified_sequence(artifact)


def test_regeneration_is_byte_identical_and_detects_wrong_config(
    tmp_path: Path, config: CausalExperimentConfig, code: CSSCode
) -> None:
    _, observed, supervision, diagnostics, manifest = _materialize(config, code)
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_sequence(first, observed, supervision, diagnostics, manifest)
    write_sequence(second, observed, supervision, diagnostics, manifest)

    for name in ("observed.npz", "supervision.npz", "diagnostics.npz", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    regenerate_and_verify(first, config, code)

    wrong = CausalExperimentConfig.from_dict(
        {**config.to_dict(), "campaign_seed": config.campaign_seed + 1}
    )
    with pytest.raises(ValueError, match="configuration hash"):
        regenerate_and_verify(first, wrong, code)
