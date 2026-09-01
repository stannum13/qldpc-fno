from __future__ import annotations

import json
from pathlib import Path

import pytest

from qldpc_fno.campaign.storage import LocalArtifactStore, materialize_completion


class RecordingLocalStore(LocalArtifactStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.uploaded_keys: list[str] = []

    def upload(self, source: Path, key: str) -> None:
        self.uploaded_keys.append(key)
        super().upload(source, key)


def test_publish_directory_writes_verified_completion_manifest_last(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "nested").mkdir()
    (source / "result.json").write_text('{"value": 7}\n')
    (source / "nested" / "weights.bin").write_bytes(b"weights")
    store = RecordingLocalStore(tmp_path / "store")

    completion = store.publish_directory(source, "evaluation")

    assert store.uploaded_keys[-1] == "evaluation/_COMPLETE.json"
    assert completion["status"] == "complete"
    assert set(completion["files"]) == {"nested/weights.bin", "result.json"}
    assert store.verify_completion("evaluation") is True
    assert json.loads((tmp_path / "store/evaluation/_COMPLETE.json").read_text()) == completion


def test_partial_files_and_corruption_never_verify_as_complete(tmp_path: Path) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"partial")
    store = LocalArtifactStore(tmp_path / "store")
    store.upload(source, "training/checkpoint.bin")

    assert store.verify_completion("training") is False

    directory = tmp_path / "complete"
    directory.mkdir()
    (directory / "checkpoint.bin").write_bytes(b"verified")
    store.publish_directory(directory, "model")
    (tmp_path / "store/model/checkpoint.bin").write_bytes(b"corrupt")

    assert store.verify_completion("model") is False


def test_corrupt_completion_is_superseded_by_immutable_recovery_generation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "artifact.bin").write_bytes(b"verified")
    store = LocalArtifactStore(tmp_path / "store")
    store.publish_directory(source, "training")
    (tmp_path / "store/training/artifact.bin").write_bytes(b"corrupt")

    recovery = store.publish_directory(source, "training")

    assert recovery["prefix"] == "training/.recovery/00000000"
    assert store.verify_completion("training") is True
    restored = tmp_path / "restored"
    materialize_completion(store, "training", restored)
    assert (restored / "artifact.bin").read_bytes() == b"verified"


def test_interrupted_recovery_generation_is_skipped_immutably(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "artifact.bin").write_bytes(b"verified")
    store = LocalArtifactStore(tmp_path / "store")
    store.publish_directory(source, "training")
    (tmp_path / "store/training/artifact.bin").write_bytes(b"corrupt")
    interrupted = tmp_path / "interrupted.bin"
    interrupted.write_bytes(b"different")
    store.upload(interrupted, "training/.recovery/00000000/artifact.bin")

    recovery = store.publish_directory(source, "training")

    assert recovery["prefix"] == "training/.recovery/00000001"
    assert store.verify_completion("training") is True


def test_interrupted_recovery_is_skipped_without_canonical_completion(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "artifact.bin").write_bytes(b"verified")
    store = LocalArtifactStore(tmp_path / "store")
    conflicting = tmp_path / "conflicting.bin"
    conflicting.write_bytes(b"canonical-conflict")
    store.upload(conflicting, "training/artifact.bin")
    interrupted = tmp_path / "interrupted.bin"
    interrupted.write_bytes(b"recovery-conflict")
    store.upload(interrupted, "training/.recovery/00000000/artifact.bin")

    recovery = store.publish_directory(source, "training")

    assert recovery["prefix"] == "training/.recovery/00000001"
    assert store.verify_completion("training") is True


def test_corrupt_recovery_advances_even_when_canonical_prefix_is_empty(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "artifact.bin").write_bytes(b"verified")
    store = LocalArtifactStore(tmp_path / "store")
    conflict = tmp_path / "conflict.bin"
    conflict.write_bytes(b"conflict")
    store.upload(conflict, "training/artifact.bin")
    first = store.publish_directory(source, "training")
    assert first["prefix"] == "training/.recovery/00000000"
    (tmp_path / "store/training/artifact.bin").unlink()
    (tmp_path / "store/training/.recovery/00000000/artifact.bin").write_bytes(b"corrupt")

    replacement = store.publish_directory(source, "training")

    assert replacement["prefix"] == "training/.recovery/00000001"
    assert store.verify_completion("training") is True


def test_publication_is_idempotent_but_never_overwrites_immutable_content(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    artifact = source / "artifact.txt"
    artifact.write_text("first")
    store = LocalArtifactStore(tmp_path / "store")

    first = store.publish_directory(source, "pilot")
    second = store.publish_directory(source, "pilot")
    assert second == first

    artifact.write_text("changed")
    with pytest.raises(FileExistsError, match="immutable"):
        store.publish_directory(source, "pilot")
    assert (tmp_path / "store/pilot/artifact.txt").read_text() == "first"


@pytest.mark.parametrize("key", ["../escape", "/absolute", "safe/../../escape", ""])
def test_store_rejects_unsafe_object_keys(tmp_path: Path, key: str) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"data")
    store = LocalArtifactStore(tmp_path / "store")

    with pytest.raises(ValueError, match="safe relative"):
        store.upload(source, key)


def test_local_store_rejects_symlink_escape_inside_store_root(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"data")
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "store"
    store = LocalArtifactStore(root)
    (root / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        store.upload(source, "linked/escaped.bin")
    assert not (outside / "escaped.bin").exists()


def test_store_rejects_symlinked_publication_source(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "artifact.bin").write_bytes(b"data")
    source = tmp_path / "source"
    source.symlink_to(outside, target_is_directory=True)
    store = LocalArtifactStore(tmp_path / "store")

    with pytest.raises(ValueError, match="source must not be a symlink"):
        store.publish_directory(source, "stage")


def test_materialization_rejects_symlink_escape_in_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "nested/artifact.bin").write_bytes(b"data")
    store = LocalArtifactStore(tmp_path / "store")
    store.publish_directory(source, "stage")
    destination = tmp_path / "destination"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination / "nested").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        materialize_completion(store, "stage", destination)
    assert not (outside / "artifact.bin").exists()
