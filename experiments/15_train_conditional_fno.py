from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn

from qldpc_fno.artifacts import sha256_file, verify_sha256, write_canonical_json
from qldpc_fno.campaign.config import CampaignConfig
from qldpc_fno.campaign.shard_io import (
    VerifiedShardSet,
    load_campaign_code,
    load_verified_shards,
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


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    write_canonical_json(temporary, payload)
    temporary.replace(path)


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
    train_stop: int,
    hx: object,
    logical_x: object,
    identity: dict[str, object],
) -> dict[str, object]:
    cache_path = output / "teacher_corrections.b8"
    metadata_path = output / "teacher.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("source") != identity:
            raise ValueError("teacher cache provenance does not match training sources")
        verify_sha256(cache_path, str(metadata["sha256"]), label="teacher correction cache")
        return metadata

    temporary = output / ".teacher_corrections.b8.tmp"
    positives = np.zeros(58, dtype=np.int64)
    with temporary.open("wb") as handle:
        for offset in range(0, shards.shots, batch_size):
            indices = np.arange(offset, min(offset + batch_size, shards.shots), dtype=np.int64)
            corrections = _teacher_batch(shards, indices, hx=hx, logical_x=logical_x)
            handle.write(np.packbits(corrections, axis=1, bitorder="little").tobytes(order="C"))
            train_rows = indices < train_stop
            if np.any(train_rows):
                positives += (
                    corrections[train_rows].reshape(-1, 58, 45).sum(axis=(0, 2), dtype=np.int64)
                )
    temporary.replace(cache_path)
    metadata = {
        "bits_per_shot": 2610,
        "decoder": _TEACHER_CONFIGURATION,
        "positive_counts_by_channel": positives.tolist(),
        "sha256": sha256_file(cache_path),
        "shots": shards.shots,
        "source": identity,
    }
    _write_json_atomic(metadata_path, metadata)
    return metadata


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


def _verify_completed_model(output: Path, identity: dict[str, object]) -> bool:
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
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-epochs-this-run", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = CampaignConfig.from_json(args.config)
    if args.max_epochs_this_run is not None and args.max_epochs_this_run <= 0:
        raise ValueError("max-epochs-this-run must be positive")
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

    if args.out.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite or implicitly resume {args.out}")
    if args.resume and not args.out.exists():
        raise FileNotFoundError("resume requested but the model output does not exist")
    args.out.mkdir(parents=True, exist_ok=True)
    if _verify_completed_model(args.out, identity):
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite completed model {args.out}")
        return

    train_stop = max(1, shards.shots * 3 // 4)
    if train_stop >= shards.shots:
        train_stop = shards.shots - 1
    teacher = _prepare_teacher_cache(
        args.out,
        shards=shards,
        batch_size=config.training_batch_size,
        train_stop=train_stop,
        hx=hx,
        logical_x=logical_x,
        identity=identity,
    )
    teacher_path = args.out / "teacher_corrections.b8"

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.training_seed)
    shuffle_rng = np.random.default_rng(config.training_seed)
    model = RingFNO(**_MODEL_CONFIGURATION)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training_learning_rate)
    start_epoch = 0
    loss_history: list[float] = []
    validation_nll_history: list[float] = []

    resume_path = args.out / "resume.json"
    if args.resume:
        if not resume_path.exists():
            raise ValueError("resume requested but resume.json is missing")
        resume = json.loads(resume_path.read_text())
        if resume.get("identity") != identity:
            raise ValueError("resume provenance does not match configuration, Git, or train data")
        checkpoint_path = args.out / str(resume["checkpoint"])
        verify_sha256(checkpoint_path, str(resume["checkpoint_sha256"]), label="checkpoint")
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if saved.get("identity") != identity or saved.get("teacher_sha256") != teacher["sha256"]:
            raise ValueError("checkpoint internal provenance does not match resume metadata")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_epoch = int(saved["epoch"])
        loss_history = list(saved["loss_history"])
        validation_nll_history = list(saved["validation_nll_history"])
        shuffle_rng.bit_generator.state = saved["rng_state"]["numpy"]
        torch.set_rng_state(saved["rng_state"]["torch"])
    elif resume_path.exists():
        raise FileExistsError("partial training state requires --resume")

    if start_epoch > config.training_epochs:
        raise ValueError("checkpoint epoch exceeds configured training_epochs")
    run_epochs = config.training_epochs - start_epoch
    if args.max_epochs_this_run is not None:
        run_epochs = min(run_epochs, args.max_epochs_this_run)
    final_epoch = start_epoch + run_epochs

    positive_counts = torch.tensor(teacher["positive_counts_by_channel"], dtype=torch.float32)
    observations = train_stop * 45
    negative_counts = observations - positive_counts
    positive_weight = (
        (negative_counts / positive_counts.clamp_min(1)).clamp(1, 100).reshape(1, -1, 1)
    )
    weighted_bce = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    validation_bce = nn.BCEWithLogitsLoss(reduction="sum")
    train_indices = np.arange(train_stop, dtype=np.int64)
    validation_indices = np.arange(train_stop, shards.shots, dtype=np.int64)

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
        loss_history.append(loss_sum / train_stop)

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
                "teacher_sha256": teacher["sha256"],
                "validation_nll_history": validation_nll_history,
            },
        )
        _write_json_atomic(
            resume_path,
            {
                "checkpoint": checkpoint_path.name,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "epoch": epoch,
                "identity": identity,
                "teacher_sha256": teacher["sha256"],
            },
        )

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
    source_sha256 = {key: value for key, value in identity.items() if key != "git_commit"}
    _write_json_atomic(
        args.out / "model.json",
        {
            "complete": True,
            "configuration": _MODEL_CONFIGURATION,
            "epoch": config.training_epochs,
            "format": "pytorch_state_dict",
            "git_commit": identity["git_commit"],
            "loss_history": loss_history,
            "sha256": model_sha256,
            "source_role": "train",
            "source_sha256": source_sha256,
            "split": {
                "policy": "contiguous_first_75_percent_with_nonempty_validation",
                "train": {"start": 0, "stop": train_stop},
                "validation": {"start": train_stop, "stop": shards.shots},
            },
            "teacher": {
                "corrections_sha256": teacher["sha256"],
                "decoder": _TEACHER_CONFIGURATION,
                "derivation": "pinned_bplsd_from_train_syndromes_at_manifest_error_rate",
                "source_role": "train",
            },
            "validation_nll_history": validation_nll_history,
        },
    )


if __name__ == "__main__":
    main()
