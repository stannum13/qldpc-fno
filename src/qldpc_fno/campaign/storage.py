"""Immutable local and Cloud Storage publications for campaign artifacts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Protocol, runtime_checkable

from google.cloud import storage as google_storage

from qldpc_fno.artifacts import sha256_file, write_canonical_json

_COMPLETION_NAME = "_COMPLETE.json"
_SCHEMA_VERSION = 1


def _safe_key(value: str, *, label: str = "object key") -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a safe relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a safe relative POSIX path")
    normalized = path.as_posix()
    if normalized != value.rstrip("/") or value.endswith("/"):
        raise ValueError(f"{label} must be a safe relative POSIX path")
    return normalized


def _join_key(prefix: str, relative: str) -> str:
    return f"{_safe_key(prefix, label='prefix')}/{_safe_key(relative)}"


def _file_record(path: Path) -> dict[str, object]:
    return {"sha256": sha256_file(path), "size": path.stat().st_size}


def _source_records(source: Path) -> tuple[dict[str, dict[str, object]], dict[str, Path]]:
    if source.is_symlink():
        raise ValueError(f"artifact publication source must not be a symlink: {source}")
    if not source.is_dir():
        raise NotADirectoryError(source)
    records: dict[str, dict[str, object]] = {}
    paths: dict[str, Path] = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact publications do not follow symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        if relative == _COMPLETION_NAME:
            raise ValueError(f"source directory must not contain {_COMPLETION_NAME}")
        _safe_key(relative, label="artifact path")
        records[relative] = _file_record(path)
        paths[relative] = path
    return records, paths


@runtime_checkable
class ArtifactStore(Protocol):
    """Storage operations required by the resumable campaign runner."""

    def exists(self, key: str) -> bool: ...

    def download(self, key: str, destination: Path) -> None: ...

    def upload(self, source: Path, key: str) -> None: ...

    def read_json(self, key: str) -> object: ...

    def publish_directory(
        self,
        source: Path,
        prefix: str,
        *,
        status: str = "complete",
        deleted: tuple[str, ...] = (),
        deadline_monotonic: float | None = None,
    ) -> dict[str, object]: ...

    def verify_completion(
        self,
        prefix: str,
        *,
        deadline_monotonic: float | None = None,
    ) -> bool: ...


class _PublishingStore:
    """Backend-independent completion-manifest-last publication logic."""

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def download(self, key: str, destination: Path) -> None:
        raise NotImplementedError

    def upload(self, source: Path, key: str) -> None:
        raise NotImplementedError

    def read_json(self, key: str) -> object:
        raise NotImplementedError

    def _copy(self, source_key: str, destination_key: str) -> None:
        raise NotImplementedError

    def _remaining_timeout(self) -> float | None:
        deadline = getattr(self, "_active_deadline_monotonic", None)
        if deadline is None:
            return None
        clock = getattr(self, "_monotonic", monotonic)
        remaining = float(deadline) - clock()
        if remaining <= 0.0:
            raise TimeoutError("artifact persistence deadline expired")
        return remaining

    @contextmanager
    def _deadline_scope(self, deadline_monotonic: float | None):
        previous = getattr(self, "_active_deadline_monotonic", None)
        if deadline_monotonic is None:
            effective = previous
        elif previous is None:
            effective = float(deadline_monotonic)
        else:
            effective = min(float(previous), float(deadline_monotonic))
        self._active_deadline_monotonic = effective
        try:
            self._remaining_timeout()
            yield
            self._remaining_timeout()
        finally:
            self._active_deadline_monotonic = previous

    def _object_matches(self, key: str, record: Mapping[str, object]) -> bool:
        size = record.get("size")
        digest = record.get("sha256")
        if type(size) is not int or size < 0 or not isinstance(digest, str):
            return False
        with tempfile.TemporaryDirectory(prefix="qldpc-fno-verify-") as temporary:
            path = Path(temporary) / "artifact"
            try:
                self.download(key, path)
            except FileNotFoundError, OSError, ValueError:
                return False
            return path.stat().st_size == size and sha256_file(path) == digest

    def _verify_exact_completion(self, prefix: str) -> bool:
        try:
            prefix = _safe_key(prefix, label="prefix")
            completion_key = f"{prefix}/{_COMPLETION_NAME}"
            if not self.exists(completion_key):
                return False
            manifest = self.read_json(completion_key)
            if not isinstance(manifest, dict):
                return False
            if (
                manifest.get("schema_version") != _SCHEMA_VERSION
                or manifest.get("prefix") != prefix
                or manifest.get("status") not in {"complete", "partial_deadline"}
            ):
                return False
            files = manifest.get("files")
            if not isinstance(files, dict):
                return False
            deleted = manifest.get("deleted", [])
            if not isinstance(deleted, list) or any(
                not isinstance(relative, str) for relative in deleted
            ):
                return False
            for relative in deleted:
                _safe_key(relative, label="deleted artifact path")
            if set(deleted) & set(files):
                return False
            for relative, record in files.items():
                if not isinstance(relative, str) or not isinstance(record, dict):
                    return False
                _safe_key(relative, label="completion artifact path")
                if not self._object_matches(_join_key(prefix, relative), record):
                    return False
            return True
        except json.JSONDecodeError, OSError, TypeError, ValueError:
            return False

    def _recovery_prefixes(self, prefix: str) -> tuple[str, ...]:
        result: list[str] = []
        for index in range(1_000_000):
            recovery = f"{prefix}/.recovery/{index:08d}"
            if not self.exists(f"{recovery}/{_COMPLETION_NAME}"):
                break
            result.append(recovery)
        return tuple(result)

    def _resolved_completion_prefix(self, prefix: str) -> str | None:
        prefix = _safe_key(prefix, label="prefix")
        recoveries = self._recovery_prefixes(prefix)
        if recoveries:
            latest = recoveries[-1]
            return latest if self._verify_exact_completion(latest) else None
        return prefix if self._verify_exact_completion(prefix) else None

    def verify_completion(
        self,
        prefix: str,
        *,
        deadline_monotonic: float | None = None,
    ) -> bool:
        with self._deadline_scope(deadline_monotonic):
            try:
                return self._resolved_completion_prefix(prefix) is not None
            except OSError, TypeError, ValueError:
                return False

    def publish_directory(
        self,
        source: Path,
        prefix: str,
        *,
        status: str = "complete",
        deleted: tuple[str, ...] = (),
        deadline_monotonic: float | None = None,
    ) -> dict[str, object]:
        with self._deadline_scope(deadline_monotonic):
            return self._publish_directory(source, prefix, status=status, deleted=deleted)

    def _publish_directory(
        self,
        source: Path,
        prefix: str,
        *,
        status: str,
        deleted: tuple[str, ...],
    ) -> dict[str, object]:
        logical_prefix = _safe_key(prefix, label="prefix")
        if status not in {"complete", "partial_deadline"}:
            raise ValueError("publication status must be complete or partial_deadline")
        records, paths = _source_records(source)
        deleted_paths = tuple(
            sorted({_safe_key(path, label="deleted artifact path") for path in deleted})
        )
        if set(deleted_paths) & set(records):
            raise ValueError("a publication cannot both delete and publish the same artifact")
        resolved = self._resolved_completion_prefix(logical_prefix)
        if resolved is not None:
            existing = self.read_json(f"{resolved}/{_COMPLETION_NAME}")
            if (
                isinstance(existing, dict)
                and existing.get("files") == records
                and existing.get("status") == status
                and tuple(existing.get("deleted", ())) == deleted_paths
            ):
                return existing
            raise FileExistsError(f"immutable completed prefix differs: {logical_prefix}")

        prefix = logical_prefix
        canonical_completion = f"{logical_prefix}/{_COMPLETION_NAME}"
        canonical_conflict = any(
            self.exists(_join_key(logical_prefix, relative))
            and not self._object_matches(_join_key(logical_prefix, relative), record)
            for relative, record in records.items()
        )
        if (
            self.exists(canonical_completion)
            or canonical_conflict
            or self._recovery_prefixes(logical_prefix)
        ):
            recovery_index = 0
            while recovery_index < 1_000_000:
                candidate = f"{logical_prefix}/.recovery/{recovery_index:08d}"
                conflicts = self.exists(f"{candidate}/{_COMPLETION_NAME}") or any(
                    self.exists(_join_key(candidate, relative))
                    and not self._object_matches(_join_key(candidate, relative), record)
                    for relative, record in records.items()
                )
                if not conflicts:
                    prefix = candidate
                    break
                recovery_index += 1
            else:
                raise RuntimeError(f"too many recovery generations: {logical_prefix}")
        completion_key = f"{prefix}/{_COMPLETION_NAME}"

        publication_id = uuid.uuid4().hex
        partial_prefix = f"{prefix}/.partial/{publication_id}"
        for relative, source_path in paths.items():
            partial_key = _join_key(partial_prefix, relative)
            self.upload(source_path, partial_key)
            if not self._object_matches(partial_key, records[relative]):
                raise OSError(f"uploaded artifact failed content verification: {partial_key}")

        for relative, record in records.items():
            destination_key = _join_key(prefix, relative)
            partial_key = _join_key(partial_prefix, relative)
            if self.exists(destination_key):
                if not self._object_matches(destination_key, record):
                    raise FileExistsError(
                        f"refusing to overwrite immutable artifact: {destination_key}"
                    )
            else:
                self._copy(partial_key, destination_key)
            if not self._object_matches(destination_key, record):
                raise OSError(f"published artifact failed content verification: {destination_key}")

        completion: dict[str, object] = {
            "deleted": list(deleted_paths),
            "files": records,
            "prefix": prefix,
            "publication_id": publication_id,
            "schema_version": _SCHEMA_VERSION,
            "status": status,
        }
        with tempfile.TemporaryDirectory(prefix="qldpc-fno-manifest-") as temporary:
            manifest_path = Path(temporary) / _COMPLETION_NAME
            write_canonical_json(manifest_path, completion)
            self.upload(manifest_path, completion_key)
        if not self._verify_exact_completion(prefix):
            raise OSError(f"completion manifest failed verification: {prefix}")
        return completion


class LocalArtifactStore(_PublishingStore):
    """Immutable artifact store rooted at a local directory."""

    def __init__(self, root: Path, *, monotonic_clock=monotonic) -> None:
        self.root = root.resolve()
        self._monotonic = monotonic_clock
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        parts = PurePosixPath(_safe_key(key)).parts
        candidate = self.root
        for part in parts:
            candidate /= part
            if candidate.is_symlink():
                raise ValueError(f"artifact path traverses a symlink: {key}")
        return candidate

    def _recovery_prefixes(self, prefix: str) -> tuple[str, ...]:
        recovery_root = self._path(f"{prefix}/.recovery")
        if not recovery_root.is_dir():
            return ()
        result: list[str] = []
        for generation in sorted(recovery_root.iterdir()):
            if generation.is_symlink() or not generation.is_dir():
                continue
            if len(generation.name) != 8 or not generation.name.isdecimal():
                continue
            completion = generation / _COMPLETION_NAME
            if completion.is_file() and not completion.is_symlink():
                result.append(f"{prefix}/.recovery/{generation.name}")
        return tuple(result)

    def exists(self, key: str) -> bool:
        self._remaining_timeout()
        return self._path(key).is_file()

    def download(self, key: str, destination: Path) -> None:
        self._remaining_timeout()
        source = self._path(key)
        if not source.is_file():
            raise FileNotFoundError(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
        self._remaining_timeout()

    def upload(self, source: Path, key: str) -> None:
        self._remaining_timeout()
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(source)
        destination = self._path(key)
        if destination.exists():
            if destination.is_file() and _file_record(destination) == _file_record(source):
                return
            raise FileExistsError(f"refusing to overwrite immutable artifact: {key}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        shutil.copyfile(source, temporary)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_file() and _file_record(destination) == _file_record(source):
                return
            raise FileExistsError(f"refusing to overwrite immutable artifact: {key}") from None
        finally:
            temporary.unlink(missing_ok=True)
        self._remaining_timeout()

    def read_json(self, key: str) -> object:
        self._remaining_timeout()
        path = self._path(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        return json.loads(path.read_text())

    def _copy(self, source_key: str, destination_key: str) -> None:
        self._remaining_timeout()
        self.upload(self._path(source_key), destination_key)


class GCSArtifactStore(_PublishingStore):
    """Immutable artifact store under one Cloud Storage bucket prefix."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        *,
        client: google_storage.Client | None = None,
        monotonic_clock=monotonic,
    ) -> None:
        if not bucket or "/" in bucket:
            raise ValueError("Cloud Storage bucket name is invalid")
        self._client = client or google_storage.Client()
        self._bucket = self._client.bucket(bucket)
        self._prefix = _safe_key(prefix, label="GCS base prefix") if prefix else ""
        self._monotonic = monotonic_clock

    @classmethod
    def from_uri(
        cls,
        uri: str,
        *,
        client: google_storage.Client | None = None,
    ) -> GCSArtifactStore:
        if not uri.startswith("gs://"):
            raise ValueError("Cloud Storage URI must start with gs://")
        bucket, separator, prefix = uri[5:].partition("/")
        return cls(bucket, prefix if separator else "", client=client)

    def _name(self, key: str) -> str:
        key = _safe_key(key)
        return f"{self._prefix}/{key}" if self._prefix else key

    def _recovery_prefixes(self, prefix: str) -> tuple[str, ...]:
        relative_root = f"{_safe_key(prefix, label='prefix')}/.recovery/"
        object_root = self._name(relative_root.rstrip("/")) + "/"
        result: list[str] = []
        timeout = self._remaining_timeout()
        for blob in self._client.list_blobs(self._bucket, prefix=object_root, timeout=timeout):
            relative = blob.name[len(object_root) :]
            generation, separator, leaf = relative.partition("/")
            if (
                separator
                and leaf == _COMPLETION_NAME
                and len(generation) == 8
                and generation.isdecimal()
            ):
                result.append(f"{prefix}/.recovery/{generation}")
        return tuple(sorted(result))

    def exists(self, key: str) -> bool:
        timeout = self._remaining_timeout()
        return bool(
            self._bucket.blob(self._name(key)).exists(client=self._client, timeout=timeout)
        )

    def download(self, key: str, destination: Path) -> None:
        blob = self._bucket.blob(self._name(key))
        timeout = self._remaining_timeout()
        if not blob.exists(client=self._client, timeout=timeout):
            raise FileNotFoundError(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        blob.download_to_filename(str(temporary), timeout=self._remaining_timeout())
        os.replace(temporary, destination)
        self._remaining_timeout()

    def upload(self, source: Path, key: str) -> None:
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(source)
        blob = self._bucket.blob(self._name(key))
        if blob.exists(client=self._client, timeout=self._remaining_timeout()):
            if self._object_matches(key, _file_record(source)):
                return
            raise FileExistsError(f"refusing to overwrite immutable artifact: {key}")
        blob.metadata = {"sha256": sha256_file(source)}
        blob.upload_from_filename(
            str(source),
            if_generation_match=0,
            timeout=self._remaining_timeout(),
        )
        self._remaining_timeout()

    def read_json(self, key: str) -> object:
        blob = self._bucket.blob(self._name(key))
        if not blob.exists(client=self._client, timeout=self._remaining_timeout()):
            raise FileNotFoundError(key)
        return json.loads(blob.download_as_bytes(timeout=self._remaining_timeout()))

    def _copy(self, source_key: str, destination_key: str) -> None:
        source = self._bucket.blob(self._name(source_key))
        self._bucket.copy_blob(
            source,
            self._bucket,
            new_name=self._name(destination_key),
            if_generation_match=0,
            timeout=self._remaining_timeout(),
        )
        self._remaining_timeout()


def open_artifact_store(location: str | Path) -> ArtifactStore:
    """Open a local path or ``gs://`` campaign store without mutating cloud state."""
    value = str(location)
    if value.startswith("gs://"):
        return GCSArtifactStore.from_uri(value)
    return LocalArtifactStore(Path(value))


def read_completion_manifest(
    store: ArtifactStore,
    prefix: str,
    *,
    deadline_monotonic: float | None = None,
) -> tuple[str, dict[str, object]]:
    """Resolve and read the latest verified immutable generation for *prefix*."""
    if isinstance(store, _PublishingStore):
        with store._deadline_scope(deadline_monotonic):
            return _read_completion_manifest(store, prefix)
    return _read_completion_manifest(store, prefix)


def _read_completion_manifest(
    store: ArtifactStore,
    prefix: str,
) -> tuple[str, dict[str, object]]:
    prefix = _safe_key(prefix, label="prefix")
    if isinstance(store, _PublishingStore):
        resolved = store._resolved_completion_prefix(prefix)
        if resolved is None:
            raise ValueError(f"cannot read unverified completion: {prefix}")
    else:
        latest_recovery: str | None = None
        for index in range(1_000_000):
            recovery = f"{prefix}/.recovery/{index:08d}"
            if not store.exists(f"{recovery}/{_COMPLETION_NAME}"):
                break
            latest_recovery = recovery
        resolved = latest_recovery or prefix
    if not store.verify_completion(resolved):
        raise ValueError(f"cannot read unverified completion: {resolved}")
    manifest = store.read_json(f"{resolved}/{_COMPLETION_NAME}")
    if not isinstance(manifest, dict):
        raise TypeError("completion manifest must be a JSON object")
    return resolved, manifest


def materialize_completion(
    store: ArtifactStore,
    prefix: str,
    destination: Path,
    *,
    replace_existing: bool = False,
    deadline_monotonic: float | None = None,
) -> None:
    """Restore a verified immutable publication without replacing different files."""
    if isinstance(store, _PublishingStore):
        with store._deadline_scope(deadline_monotonic):
            _materialize_completion(
                store,
                prefix,
                destination,
                replace_existing=replace_existing,
            )
            return
    _materialize_completion(store, prefix, destination, replace_existing=replace_existing)


def _materialize_completion(
    store: ArtifactStore,
    prefix: str,
    destination: Path,
    *,
    replace_existing: bool,
) -> None:
    prefix = _safe_key(prefix, label="prefix")
    resolved, manifest = read_completion_manifest(store, prefix)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise TypeError("completion manifest file table is malformed")
    if destination.is_symlink():
        raise ValueError(f"materialization destination is a symlink: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    deleted = manifest.get("deleted", [])
    if not isinstance(deleted, list) or any(not isinstance(path, str) for path in deleted):
        raise TypeError("completion manifest deleted table is malformed")
    for relative in deleted:
        relative = _safe_key(relative, label="deleted artifact path")
        target = destination
        for part in PurePosixPath(relative).parts:
            target /= part
            if target.is_symlink():
                raise ValueError(f"materialization path traverses a symlink: {target}")
        if target.exists():
            if not replace_existing or not target.is_file():
                raise FileExistsError(f"cannot apply immutable artifact deletion: {target}")
            target.unlink()
    for relative, record in manifest["files"].items():
        if not isinstance(relative, str) or not isinstance(record, dict):
            raise TypeError("completion manifest file record is malformed")
        relative = _safe_key(relative, label="completion artifact path")
        target = destination
        for part in PurePosixPath(relative).parts:
            target /= part
            if target.is_symlink():
                raise ValueError(f"materialization path traverses a symlink: {target}")
        if target.exists():
            if target.is_file() and _file_record(target) == record:
                continue
            if not replace_existing or not target.is_file():
                raise FileExistsError(f"local artifact differs from immutable store: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        store.download(_join_key(resolved, relative), target)
        if _file_record(target) != record:
            raise OSError(f"downloaded artifact failed verification: {relative}")
