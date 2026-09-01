from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from qldpc_fno.artifacts import sha256_file, verify_sha256, write_canonical_json
from qldpc_fno.campaign.config import CampaignConfig
from qldpc_fno.campaign.local import resolve_git_commit
from qldpc_fno.campaign.shard_io import (
    VerifiedShardSet,
    deterministic_stratified_split,
    load_campaign_code,
    load_verified_shards,
    load_verified_teacher_artifact,
    verify_teacher_chunk,
    verify_teacher_chunks,
)
from qldpc_fno.data.conditional_fields import add_noise_channel
from qldpc_fno.data.ring_fields import to_ring_field
from qldpc_fno.decoders.bplsd import decode_bplsd_batch
from qldpc_fno.models.fno1d import RingFNO
from qldpc_fno.stim.b8 import read_b8_rows

_MODEL_CONFIGURATION = {
    "depth": 2,
    "in_channels": 22,
    "modes": 12,
    "out_channels": 58,
    "width": 32,
}
_TEACHER_CONFIGURATION = {
    "bp_method": "minimum_sum",
    "lsd_method": "LSD_E",
    "lsd_order": 5,
    "max_iter": 100,
    "ms_scaling_factor": 0.0,
    "schedule": "serial",
}
_TEACHER_BITS_PER_SHOT = 2610
_MAX_TEACHER_CHUNK_SHOTS = 2_048


@dataclass(frozen=True, slots=True)
class _TeacherPreparation:
    metadata: dict[str, object] | None
    metadata_sha256: str | None
    progress_sha256: str


def _git_commit() -> str:
    return resolve_git_commit(Path(__file__).resolve().parents[1])


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    write_canonical_json(temporary, payload)
    temporary.replace(path)


def _publish_initialized_output(output: Path, payload: dict[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-initializing-", dir=output.parent))
    try:
        write_canonical_json(staging / "resume.json", payload)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _identity(
    *, config_path: Path, code_manifest_path: Path, shards: VerifiedShardSet
) -> dict[str, object]:
    return {
        "code_manifest": sha256_file(code_manifest_path),
        "config": sha256_file(config_path),
        "git_commit": _git_commit(),
        "train_manifest": shards.manifest_sha256,
        "train_shard_manifests": shards.shard_manifest_sha256,
    }


def _teacher_batch(
    shards: VerifiedShardSet,
    indices: np.ndarray,
    *,
    hx: object,
    logical_x: object,
) -> np.ndarray:
    syndromes = shards.read("dets.b8", indices)
    rates = shards.error_rates(indices)
    corrections = np.empty((indices.size, 2610), dtype=np.uint8)
    for rate in np.unique(rates):
        positions = np.flatnonzero(rates == rate)
        decoded = decode_bplsd_batch(
            hx,
            syndromes[positions],
            logical_x,
            error_rate=float(rate),
        )
        if not np.all(decoded.syndrome_valid):
            raise RuntimeError("pinned BP-LSD teacher produced a syndrome-invalid correction")
        corrections[positions] = decoded.corrections
    return corrections


def _prepare_teacher_cache(
    output: Path,
    *,
    shards: VerifiedShardSet,
    batch_size: int,
    train_indices: np.ndarray,
    hx: object,
    logical_x: object,
    identity: dict[str, object],
    expected_metadata_sha256: str | None,
    max_new_chunks: int | None,
) -> _TeacherPreparation:
    cache_path = output / "teacher_corrections.b8"
    metadata_path = output / "teacher.json"
    chunk_shots = min(batch_size, _MAX_TEACHER_CHUNK_SHOTS)
    chunk_dir = output / "teacher_chunks"
    progress_path = output / "teacher_progress.json"
    if metadata_path.exists():
        metadata = load_verified_teacher_artifact(
            output,
            expected_metadata_sha256=expected_metadata_sha256,
            expected_shots=shards.shots,
        )
        if metadata.get("source") != identity:
            raise ValueError("teacher cache provenance does not match training sources")
        if metadata.get("decoder") != _TEACHER_CONFIGURATION:
            raise ValueError("teacher cache decoder configuration mismatch")
        if metadata.get("chunk_shots") != chunk_shots:
            raise ValueError("teacher chunk cadence does not match training configuration")
        verified_chunks = metadata["chunks"]
        if not isinstance(verified_chunks, dict):
            raise TypeError("teacher chunk provenance must be an object")
        positives = _teacher_positive_counts(
            cache_path,
            shards=shards,
            train_indices=train_indices,
            batch_size=batch_size,
        )
        if metadata.get("positive_counts_by_channel") != positives.tolist():
            raise ValueError("teacher positive counts do not match packed corrections")
        metadata_sha256 = sha256_file(metadata_path)
        progress_sha256 = _write_teacher_progress(
            progress_path,
            chunks=verified_chunks,
            chunk_shots=chunk_shots,
            complete=True,
            identity=identity,
            shots=shards.shots,
            teacher_metadata_sha256=metadata_sha256,
        )
        return _TeacherPreparation(metadata, metadata_sha256, progress_sha256)

    chunk_dir.mkdir(exist_ok=True)
    expected_chunk_count = (shards.shots + chunk_shots - 1) // chunk_shots
    expected_manifest_names = {f"chunk-{index:05d}.json" for index in range(expected_chunk_count)}
    discovered_manifest_names = {path.name for path in chunk_dir.glob("chunk-*.json")}
    unexpected = discovered_manifest_names - expected_manifest_names
    if unexpected:
        raise ValueError(f"unexpected teacher chunk manifests: {sorted(unexpected)}")

    chunks: dict[str, str] = {}
    generated = 0
    for chunk_index in range(expected_chunk_count):
        start = chunk_index * chunk_shots
        stop = min(start + chunk_shots, shards.shots)
        manifest_path = chunk_dir / f"chunk-{chunk_index:05d}.json"
        if manifest_path.exists():
            verify_teacher_chunk(
                manifest_path,
                chunk_index=chunk_index,
                start=start,
                stop=stop,
                source=identity,
                decoder=_TEACHER_CONFIGURATION,
            )
        else:
            if max_new_chunks is not None and generated >= max_new_chunks:
                progress_sha256 = _write_teacher_progress(
                    progress_path,
                    chunks=chunks,
                    chunk_shots=chunk_shots,
                    complete=False,
                    identity=identity,
                    shots=shards.shots,
                )
                return _TeacherPreparation(None, None, progress_sha256)
            indices = np.arange(start, stop, dtype=np.int64)
            corrections = _teacher_batch(shards, indices, hx=hx, logical_x=logical_x)
            packed = np.packbits(corrections, axis=1, bitorder="little").tobytes(order="C")
            chunk_path = chunk_dir / f"chunk-{chunk_index:05d}.b8"
            temporary_chunk = chunk_dir / f".chunk-{chunk_index:05d}.b8.tmp"
            temporary_chunk.write_bytes(packed)
            os.replace(temporary_chunk, chunk_path)
            _write_json_atomic(
                manifest_path,
                {
                    "bits_per_shot": _TEACHER_BITS_PER_SHOT,
                    "chunk_index": chunk_index,
                    "decoder": _TEACHER_CONFIGURATION,
                    "path": chunk_path.name,
                    "sha256": sha256_file(chunk_path),
                    "shots": stop - start,
                    "source": identity,
                    "start": start,
                    "stop": stop,
                },
            )
            generated += 1
        chunks[str(manifest_path.relative_to(output))] = sha256_file(manifest_path)
        _write_teacher_progress(
            progress_path,
            chunks=chunks,
            chunk_shots=chunk_shots,
            complete=False,
            identity=identity,
            shots=shards.shots,
        )

    verified_chunks = verify_teacher_chunks(
        output,
        expected_shots=shards.shots,
        chunk_shots=chunk_shots,
        source=identity,
        decoder=_TEACHER_CONFIGURATION,
    )
    if chunks != verified_chunks:
        raise ValueError("teacher chunk set changed before final assembly")
    temporary = output / ".teacher_corrections.b8.tmp"
    with temporary.open("wb") as handle:
        for chunk_index in range(expected_chunk_count):
            handle.write((chunk_dir / f"chunk-{chunk_index:05d}.b8").read_bytes())
    expected_size = shards.shots * ((_TEACHER_BITS_PER_SHOT + 7) // 8)
    if temporary.stat().st_size != expected_size:
        raise ValueError("assembled teacher correction cache has the wrong size")
    os.replace(temporary, cache_path)
    positives = _teacher_positive_counts(
        cache_path,
        shards=shards,
        train_indices=train_indices,
        batch_size=batch_size,
    )
    metadata = {
        "bits_per_shot": _TEACHER_BITS_PER_SHOT,
        "chunk_shots": chunk_shots,
        "chunks": chunks,
        "decoder": _TEACHER_CONFIGURATION,
        "positive_counts_by_channel": positives.tolist(),
        "sha256": sha256_file(cache_path),
        "shots": shards.shots,
        "source": identity,
    }
    _write_json_atomic(metadata_path, metadata)
    metadata_sha256 = sha256_file(metadata_path)
    progress_sha256 = _write_teacher_progress(
        progress_path,
        chunks=chunks,
        chunk_shots=chunk_shots,
        complete=True,
        identity=identity,
        shots=shards.shots,
        teacher_metadata_sha256=metadata_sha256,
    )
    return _TeacherPreparation(metadata, metadata_sha256, progress_sha256)


def _write_teacher_progress(
    path: Path,
    *,
    chunks: dict[str, str],
    chunk_shots: int,
    complete: bool,
    identity: dict[str, object],
    shots: int,
    teacher_metadata_sha256: str | None = None,
) -> str:
    _write_json_atomic(
        path,
        {
            "chunk_shots": chunk_shots,
            "chunks": chunks,
            "complete": complete,
            "shots": shots,
            "source": identity,
            "teacher_metadata_sha256": teacher_metadata_sha256,
        },
    )
    return sha256_file(path)


def _teacher_positive_counts(
    cache_path: Path,
    *,
    shards: VerifiedShardSet,
    train_indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    positives = np.zeros(58, dtype=np.int64)
    for offset in range(0, train_indices.size, batch_size):
        indices = train_indices[offset : offset + batch_size]
        corrections = read_b8_rows(
            cache_path,
            rows=indices,
            shots=shards.shots,
            bits_per_shot=2610,
        )
        positives += corrections.reshape(-1, 58, 45).sum(axis=(0, 2), dtype=np.int64)
    return positives


def _load_training_batch(
    shards: VerifiedShardSet,
    teacher_path: Path,
    indices: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    syndromes = to_ring_field(shards.read("dets.b8", indices), channels=21, ell=45)
    inputs = add_noise_channel(syndromes, shards.error_rates(indices))
    correction_bits = read_b8_rows(
        teacher_path,
        rows=indices,
        shots=shards.shots,
        bits_per_shot=2610,
    )
    targets = to_ring_field(correction_bits, channels=58, ell=45)
    return torch.from_numpy(inputs), torch.from_numpy(targets)


def _save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _verify_completed_model(
    output: Path,
    identity: dict[str, object],
    *,
    expected_shots: int,
) -> bool:
    manifest_path = output / "model.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("complete") is not True or manifest.get("source_role") != "train":
        raise ValueError("existing model publication is not complete training output")
    if manifest.get("git_commit") != identity["git_commit"]:
        raise ValueError("model Git commit does not match the current training run")
    expected_sources = {key: value for key, value in identity.items() if key != "git_commit"}
    if manifest.get("source_sha256") != expected_sources:
        raise ValueError("model source provenance does not match the current training run")
    verify_sha256(output / "model.pt", str(manifest["sha256"]), label="model")
    teacher = manifest.get("teacher")
    if not isinstance(teacher, dict):
        raise TypeError("completed model is missing teacher provenance")
    teacher_metadata = load_verified_teacher_artifact(
        output,
        expected_metadata_sha256=str(teacher["metadata_sha256"]),
        expected_shots=expected_shots,
    )
    if teacher_metadata.get("sha256") != teacher.get("corrections_sha256"):
        raise ValueError("model teacher correction SHA-256 disagrees with teacher metadata")
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("completed model is missing epoch checkpoint candidates")
    for checkpoint in checkpoints:
        verify_sha256(
            output / str(checkpoint["path"]),
            str(checkpoint["sha256"]),
            label=f"epoch {checkpoint['epoch']} checkpoint",
        )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--prepare-teacher-only", action="store_true")
    parser.add_argument("--max-teacher-chunks-this-run", type=int)
    parser.add_argument("--max-epochs-this-run", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = CampaignConfig.from_json(args.config)
    if args.max_epochs_this_run is not None and args.max_epochs_this_run <= 0:
        raise ValueError("max-epochs-this-run must be positive")
    if args.max_teacher_chunks_this_run is not None and args.max_teacher_chunks_this_run <= 0:
        raise ValueError("max-teacher-chunks-this-run must be positive")
    if args.initialize_only and args.prepare_teacher_only:
        raise ValueError("initialize-only and prepare-teacher-only are mutually exclusive")
    code_manifest_path = args.code / "code.json"
    _, hx, _, logical_x = load_campaign_code(args.code)
    shards = load_verified_shards(
        args.train,
        role="train",
        config_path=args.config,
        code_manifest_path=code_manifest_path,
    )
    if shards.shots < 2:
        raise ValueError("conditional training requires at least two train-role shots")
    identity = _identity(
        config_path=args.config, code_manifest_path=code_manifest_path, shards=shards
    )
    train_indices, validation_indices, split_metadata = deterministic_stratified_split(
        shards,
        training_seed=config.training_seed,
    )

    if args.out.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite or implicitly resume {args.out}")
    if args.resume and not args.out.exists():
        raise FileNotFoundError("resume requested but the model output does not exist")
    if args.resume and _verify_completed_model(
        args.out,
        identity,
        expected_shots=shards.shots,
    ):
        return

    resume_path = args.out / "resume.json"
    if args.resume:
        if not resume_path.exists():
            raise ValueError("resume requested but resume.json is missing")
        resume = json.loads(resume_path.read_text())
        if resume.get("identity") != identity:
            raise ValueError("resume provenance does not match configuration, Git, or train data")
        if resume.get("split") != split_metadata:
            raise ValueError("resume split provenance does not match deterministic stratification")
    else:
        resume = {
            "checkpoint": None,
            "checkpoint_sha256": None,
            "epoch": 0,
            "identity": identity,
            "split": split_metadata,
            "status": "initialized",
            "teacher_corrections_sha256": None,
            "teacher_metadata_sha256": None,
            "teacher_progress_sha256": None,
        }
        _publish_initialized_output(args.out, resume)
    if args.initialize_only:
        return

    expected_teacher_metadata = resume.get("teacher_metadata_sha256")
    if expected_teacher_metadata is not None and not isinstance(expected_teacher_metadata, str):
        raise TypeError("resume teacher_metadata_sha256 must be a string or null")
    preparation = _prepare_teacher_cache(
        args.out,
        shards=shards,
        batch_size=config.training_batch_size,
        train_indices=train_indices,
        hx=hx,
        logical_x=logical_x,
        identity=identity,
        expected_metadata_sha256=expected_teacher_metadata,
        max_new_chunks=args.max_teacher_chunks_this_run,
    )
    if preparation.metadata is None:
        _write_json_atomic(
            resume_path,
            {
                **resume,
                "status": "teacher_partial",
                "teacher_progress_sha256": preparation.progress_sha256,
            },
        )
        return
    teacher = preparation.metadata
    teacher_metadata_sha256 = preparation.metadata_sha256
    if teacher_metadata_sha256 is None:
        raise RuntimeError("completed teacher preparation is missing its metadata hash")
    teacher_path = args.out / "teacher_corrections.b8"
    resume = {
        **resume,
        "status": "checkpointed" if resume.get("checkpoint") is not None else "teacher_ready",
        "teacher_corrections_sha256": teacher["sha256"],
        "teacher_metadata_sha256": teacher_metadata_sha256,
        "teacher_progress_sha256": preparation.progress_sha256,
    }
    _write_json_atomic(resume_path, resume)
    if args.prepare_teacher_only:
        return

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.training_seed)
    shuffle_rng = np.random.default_rng(config.training_seed)
    model = RingFNO(**_MODEL_CONFIGURATION)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training_learning_rate)
    start_epoch = 0
    loss_history: list[float] = []
    validation_nll_history: list[float] = []

    if resume.get("checkpoint") is not None:
        checkpoint_path = args.out / str(resume["checkpoint"])
        verify_sha256(checkpoint_path, str(resume["checkpoint_sha256"]), label="checkpoint")
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            saved.get("identity") != identity
            or saved.get("split") != split_metadata
            or saved.get("teacher_sha256") != teacher["sha256"]
            or saved.get("teacher_metadata_sha256") != teacher_metadata_sha256
        ):
            raise ValueError("checkpoint internal provenance does not match resume metadata")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_epoch = int(saved["epoch"])
        loss_history = list(saved["loss_history"])
        validation_nll_history = list(saved["validation_nll_history"])
        shuffle_rng.bit_generator.state = saved["rng_state"]["numpy"]
        torch.set_rng_state(saved["rng_state"]["torch"])
    if start_epoch > config.training_epochs:
        raise ValueError("checkpoint epoch exceeds configured training_epochs")
    run_epochs = config.training_epochs - start_epoch
    if args.max_epochs_this_run is not None:
        run_epochs = min(run_epochs, args.max_epochs_this_run)
    final_epoch = start_epoch + run_epochs

    positive_counts = torch.tensor(teacher["positive_counts_by_channel"], dtype=torch.float32)
    observations = train_indices.size * 45
    negative_counts = observations - positive_counts
    positive_weight = (
        (negative_counts / positive_counts.clamp_min(1)).clamp(1, 100).reshape(1, -1, 1)
    )
    weighted_bce = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    validation_bce = nn.BCEWithLogitsLoss(reduction="sum")
    for epoch in range(start_epoch + 1, final_epoch + 1):
        model.train()
        shuffled = shuffle_rng.permutation(train_indices)
        loss_sum = 0.0
        for offset in range(0, shuffled.size, config.training_batch_size):
            indices = shuffled[offset : offset + config.training_batch_size]
            inputs, targets = _load_training_batch(shards, teacher_path, indices)
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_bce(model(inputs), targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * indices.size
        loss_history.append(loss_sum / train_indices.size)

        model.eval()
        validation_sum = 0.0
        with torch.no_grad():
            for offset in range(0, validation_indices.size, config.training_batch_size):
                indices = validation_indices[offset : offset + config.training_batch_size]
                inputs, targets = _load_training_batch(shards, teacher_path, indices)
                validation_sum += float(validation_bce(model(inputs), targets).item())
        validation_nll_history.append(validation_sum / (validation_indices.size * 2610))
        rng_state = {
            "numpy": copy.deepcopy(shuffle_rng.bit_generator.state),
            "torch": torch.get_rng_state().clone(),
        }
        checkpoint_path = args.out / f"epoch-{epoch:04d}.pt"
        _save_checkpoint(
            checkpoint_path,
            {
                "epoch": epoch,
                "identity": identity,
                "loss_history": loss_history,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "rng_state": rng_state,
                "split": split_metadata,
                "teacher_metadata_sha256": teacher_metadata_sha256,
                "teacher_sha256": teacher["sha256"],
                "validation_nll_history": validation_nll_history,
            },
        )
        resume = {
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "epoch": epoch,
            "identity": identity,
            "split": split_metadata,
            "status": "checkpointed",
            "teacher_corrections_sha256": teacher["sha256"],
            "teacher_metadata_sha256": teacher_metadata_sha256,
            "teacher_progress_sha256": preparation.progress_sha256,
        }
        _write_json_atomic(resume_path, resume)

    if final_epoch < config.training_epochs:
        return
    model_path = args.out / "model.pt"
    temporary_model = args.out / ".model.pt.tmp"
    torch.save(
        {"configuration": _MODEL_CONFIGURATION, "state_dict": model.state_dict()},
        temporary_model,
    )
    temporary_model.replace(model_path)
    model_sha256 = sha256_file(model_path)
    verify_sha256(model_path, model_sha256, label="final model")
    checkpoint_candidates = []
    for epoch in range(1, config.training_epochs + 1):
        checkpoint_path = args.out / f"epoch-{epoch:04d}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing epoch checkpoint candidate: {checkpoint_path}")
        checkpoint_candidates.append(
            {
                "epoch": epoch,
                "path": checkpoint_path.name,
                "sha256": sha256_file(checkpoint_path),
            }
        )
    source_sha256 = {key: value for key, value in identity.items() if key != "git_commit"}
    model_manifest_path = args.out / "model.json"
    _write_json_atomic(
        model_manifest_path,
        {
            "complete": True,
            "checkpoints": checkpoint_candidates,
            "configuration": _MODEL_CONFIGURATION,
            "epoch": config.training_epochs,
            "format": "pytorch_state_dict",
            "git_commit": identity["git_commit"],
            "loss_history": loss_history,
            "sha256": model_sha256,
            "source_role": "train",
            "source_sha256": source_sha256,
            "split": split_metadata,
            "teacher": {
                "corrections_sha256": teacher["sha256"],
                "decoder": _TEACHER_CONFIGURATION,
                "derivation": "pinned_bplsd_from_train_syndromes_at_manifest_error_rate",
                "metadata_sha256": teacher_metadata_sha256,
                "source_role": "train",
            },
            "validation_nll_history": validation_nll_history,
        },
    )
    _write_json_atomic(
        resume_path,
        {
            **resume,
            "model_manifest_sha256": sha256_file(model_manifest_path),
            "status": "complete",
        },
    )


if __name__ == "__main__":
    main()
