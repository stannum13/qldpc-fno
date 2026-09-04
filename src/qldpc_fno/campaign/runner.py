"""Resumable, deadline-aware orchestration for the accuracy campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from time import monotonic

from qldpc_fno.artifacts import sha256_file, verify_sha256, write_canonical_json
from qldpc_fno.campaign.config import CampaignConfig
from qldpc_fno.campaign.inputs import CampaignInputRequest, prepare_campaign_inputs
from qldpc_fno.campaign.local import resolve_git_commit
from qldpc_fno.campaign.shard_io import load_campaign_code, load_verified_shards
from qldpc_fno.campaign.storage import (
    ArtifactStore,
    materialize_completion,
    open_artifact_store,
    read_completion_manifest,
)

_STAGE_NAMES = ("pilot", "shards", "training", "calibration", "evaluation", "summary")


def _safe_artifact_path(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a safe relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ValueError(f"{label} must be a safe relative path")
    if root.is_symlink():
        raise ValueError(f"{label} root must not be a symlink")
    candidate = root
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink")
    return candidate


class CampaignStatus(StrEnum):
    """Terminal campaign-runner states."""

    COMPLETE = "complete"
    PARTIAL_DEADLINE = "partial_deadline"


class StageResult(StrEnum):
    """Result of one bounded stage work unit."""

    COMPLETE = "complete"
    CHECKPOINTED = "checkpointed"
    PARTIAL_DEADLINE = "partial_deadline"


@dataclass(frozen=True, slots=True)
class CampaignStage:
    """One resumable stage with an immutable publication prefix."""

    name: str
    directory: Path
    run_unit: Callable[[float | None], StageResult]

    def __post_init__(self) -> None:
        if self.name not in _STAGE_NAMES:
            raise ValueError(f"unknown campaign stage: {self.name}")


class CampaignRunner:
    """Run bounded stage units, restoring and publishing verified snapshots."""

    def __init__(
        self,
        *,
        store: ArtifactStore,
        stages: Sequence[CampaignStage],
        checkpoint_grace_seconds: float,
        monotonic: Callable[[], float] = monotonic,
        on_deadline: Callable[[str, float | None], None] | None = None,
    ) -> None:
        if checkpoint_grace_seconds < 0:
            raise ValueError("checkpoint_grace_seconds must be non-negative")
        names = tuple(stage.name for stage in stages)
        if len(set(names)) != len(names):
            raise ValueError("campaign stages must have unique names")
        order = {name: index for index, name in enumerate(_STAGE_NAMES)}
        if any(order[left] >= order[right] for left, right in pairwise(names)):
            raise ValueError("campaign stages must follow the declared stage order")
        self._store = store
        self._stages = tuple(stages)
        self._grace = float(checkpoint_grace_seconds)
        self._monotonic = monotonic
        self._on_deadline = on_deadline

    def _checkpoint_count(
        self,
        stage: CampaignStage,
        deadline_monotonic: float | None = None,
    ) -> int:
        count = 0
        while count < 1_000_000:
            prefix = f".checkpoints/{stage.name}/{count:08d}"
            completion_key = f"{prefix}/_COMPLETE.json"
            if not self._store.exists(
                completion_key,
                deadline_monotonic=deadline_monotonic,
            ) and not self._store.verify_completion(
                prefix,
                deadline_monotonic=deadline_monotonic,
            ):
                return count
            if not self._store.verify_completion(
                prefix,
                deadline_monotonic=deadline_monotonic,
            ):
                raise ValueError(f"checkpoint completion is corrupt: {prefix}")
            count += 1
        raise RuntimeError(f"too many checkpoint snapshots for stage {stage.name}")

    def _restore_checkpoints(
        self,
        stage: CampaignStage,
        count: int,
        deadline_monotonic: float | None = None,
    ) -> None:
        for index in range(count):
            materialize_completion(
                self._store,
                f".checkpoints/{stage.name}/{index:08d}",
                stage.directory,
                replace_existing=index > 0,
                deadline_monotonic=deadline_monotonic,
            )

    def _checkpoint_records(
        self,
        stage: CampaignStage,
        count: int,
        deadline_monotonic: float | None = None,
    ) -> dict[str, object]:
        records: dict[str, object] = {}
        for index in range(count):
            _, manifest = read_completion_manifest(
                self._store,
                f".checkpoints/{stage.name}/{index:08d}",
                deadline_monotonic=deadline_monotonic,
            )
            files = manifest.get("files")
            if not isinstance(files, dict):
                raise TypeError("checkpoint completion file table is malformed")
            deleted = manifest.get("deleted", [])
            if not isinstance(deleted, list) or any(not isinstance(path, str) for path in deleted):
                raise TypeError("checkpoint completion deleted table is malformed")
            for relative in deleted:
                records.pop(relative, None)
            records.update(files)
        return records

    def _publish_checkpoint(
        self,
        stage: CampaignStage,
        count: int,
        deadline_monotonic: float | None,
    ) -> None:
        previous = self._checkpoint_records(stage, count, deadline_monotonic)
        current_relatives = {
            path.relative_to(stage.directory).as_posix()
            for path in stage.directory.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        deleted = set(previous) - current_relatives
        with tempfile.TemporaryDirectory(prefix=f"qldpc-fno-{stage.name}-delta-") as temporary:
            delta = Path(temporary)
            for source in sorted(stage.directory.rglob("*")):
                if source.is_symlink():
                    raise ValueError(f"checkpoint source contains a symlink: {source}")
                if not source.is_file():
                    continue
                relative = source.relative_to(stage.directory).as_posix()
                record = {"sha256": sha256_file(source), "size": source.stat().st_size}
                if previous.get(relative) == record:
                    continue
                destination = delta / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copyfile(source, destination)
            self._store.publish_directory(
                delta,
                f".checkpoints/{stage.name}/{count:08d}",
                deleted=tuple(sorted(deleted)),
                deadline_monotonic=deadline_monotonic,
            )

    def _checkpoint_changed(
        self,
        stage: CampaignStage,
        count: int,
        deadline_monotonic: float | None = None,
    ) -> bool:
        previous = self._checkpoint_records(stage, count, deadline_monotonic)
        current = {
            path.relative_to(stage.directory).as_posix(): {
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
            for path in stage.directory.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        return previous != current

    def _deadline(
        self,
        reason: str,
        stage: CampaignStage,
        checkpoint_count: int,
        deadline_monotonic: float | None,
    ) -> CampaignStatus:
        if self._on_deadline is not None:
            try:
                self._on_deadline(reason, deadline_monotonic)
            except (OSError, TimeoutError):
                pass
        try:
            if stage.directory.is_dir() and self._checkpoint_changed(
                stage,
                checkpoint_count,
                deadline_monotonic,
            ):
                self._publish_checkpoint(stage, checkpoint_count, deadline_monotonic)
        except (OSError, TimeoutError):
            pass
        return CampaignStatus.PARTIAL_DEADLINE

    def run(self, deadline_monotonic: float | None) -> CampaignStatus:
        """Run until all stages verify or the checkpoint grace window begins."""
        if deadline_monotonic is not None:
            if type(deadline_monotonic) not in (int, float):
                raise TypeError("deadline_monotonic must be numeric or None")
            if not math.isfinite(deadline_monotonic):
                raise ValueError("deadline_monotonic must be finite")
        cutoff = None if deadline_monotonic is None else float(deadline_monotonic) - self._grace
        for stage in self._stages:
            if self._store.verify_completion(
                stage.name,
                deadline_monotonic=deadline_monotonic,
            ):
                materialize_completion(
                    self._store,
                    stage.name,
                    stage.directory,
                    deadline_monotonic=deadline_monotonic,
                )
                _, manifest = read_completion_manifest(
                    self._store,
                    stage.name,
                    deadline_monotonic=deadline_monotonic,
                )
                if isinstance(manifest, dict) and manifest.get("status") == "partial_deadline":
                    return CampaignStatus.PARTIAL_DEADLINE
                continue

            checkpoint_count = self._checkpoint_count(stage, deadline_monotonic)
            self._restore_checkpoints(stage, checkpoint_count, deadline_monotonic)
            while True:
                if cutoff is not None and self._monotonic() >= cutoff:
                    return self._deadline(
                        f"deadline_before_{stage.name}_unit",
                        stage,
                        checkpoint_count,
                        deadline_monotonic,
                    )
                try:
                    result = stage.run_unit(cutoff)
                except subprocess.TimeoutExpired:
                    return self._deadline(
                        f"deadline_during_{stage.name}_unit",
                        stage,
                        checkpoint_count,
                        deadline_monotonic,
                    )
                if result is StageResult.COMPLETE:
                    self._store.publish_directory(
                        stage.directory,
                        stage.name,
                        deadline_monotonic=deadline_monotonic,
                    )
                    break
                if result is StageResult.PARTIAL_DEADLINE:
                    return self._deadline(
                        f"{stage.name}_requested_deadline",
                        stage,
                        checkpoint_count,
                        deadline_monotonic,
                    )
                if result is not StageResult.CHECKPOINTED:
                    raise ValueError(f"stage {stage.name} returned an invalid result: {result!r}")
                self._publish_checkpoint(stage, checkpoint_count, deadline_monotonic)
                checkpoint_count += 1
        return CampaignStatus.COMPLETE


def _verified_pilot(campaign: Path) -> dict[str, object] | None:
    pilot = campaign / "pilot"
    manifest_path = pilot / "manifest.json"
    selection_path = pilot / "selection.json"
    if not manifest_path.is_file() or not selection_path.is_file():
        return None
    manifest_path = _safe_artifact_path(pilot, "manifest.json", label="pilot manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("complete") is not True or manifest.get("role") != "pilot":
        raise ValueError("pilot manifest is not a completed pilot publication")
    selection_path = _safe_artifact_path(pilot, "selection.json", label="pilot selection")
    verify_sha256(selection_path, str(manifest["selection_sha256"]), label="pilot selection")
    selection = json.loads(selection_path.read_text())
    rates = selection.get("selected_noise_points")
    if not isinstance(rates, list) or any(type(rate) not in (int, float) for rate in rates):
        raise ValueError("pilot selection noise points are malformed")
    return {
        "evidence_role": "selection_only_not_held_out",
        "manifest_sha256": sha256_file(manifest_path),
        "selected_noise_points": rates,
        "selection_sha256": sha256_file(selection_path),
    }


def _verified_model(campaign: Path) -> dict[str, object] | None:
    model = campaign / "model"
    manifest_path = model / "model.json"
    if not manifest_path.is_file():
        return None
    manifest_path = _safe_artifact_path(model, "model.json", label="model manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("complete") is not True or manifest.get("source_role") != "train":
        raise ValueError("model manifest is not a completed training publication")
    model_path = _safe_artifact_path(model, "model.pt", label="campaign model")
    verify_sha256(model_path, str(manifest["sha256"]), label="campaign model")
    for checkpoint in manifest.get("checkpoints", []):
        if not isinstance(checkpoint, dict):
            raise TypeError("model checkpoint record must be an object")
        verify_sha256(
            _safe_artifact_path(model, checkpoint.get("path"), label="model checkpoint path"),
            str(checkpoint["sha256"]),
            label="model checkpoint",
        )
    return {
        "epoch": manifest.get("epoch"),
        "git_commit": manifest.get("git_commit"),
        "manifest_sha256": sha256_file(manifest_path),
        "model_sha256": manifest["sha256"],
    }


def _verified_calibration(campaign: Path) -> dict[str, object] | None:
    calibration = campaign / "calibration"
    selected_path = calibration / "selected.json"
    grid_path = calibration / "grid.json"
    if not selected_path.is_file() or not grid_path.is_file():
        return None
    selected_path = _safe_artifact_path(calibration, "selected.json", label="calibration selection")
    grid_path = _safe_artifact_path(calibration, "grid.json", label="calibration grid")
    selected = json.loads(selected_path.read_text())
    if selected.get("complete") is not True or selected.get("source_role") != "calibration":
        raise ValueError("selected calibration is not a completed calibration publication")
    sources = selected.get("source_sha256")
    if not isinstance(sources, dict) or sources.get("grid") != sha256_file(grid_path):
        raise ValueError("selected calibration grid provenance mismatch")
    return {
        "evidence_role": "tuning_only_not_held_out",
        "grid_sha256": sha256_file(grid_path),
        "selected": selected.get("selected"),
        "selected_sha256": sha256_file(selected_path),
    }


def _verified_evaluation(campaign: Path) -> tuple[str | None, list[dict[str, object]]]:
    evaluation = campaign / "evaluation"
    manifest_path = evaluation / "manifest.json"
    if not manifest_path.is_file():
        return None, []
    manifest_path = _safe_artifact_path(evaluation, "manifest.json", label="evaluation manifest")
    manifest = json.loads(manifest_path.read_text())
    status = manifest.get("status")
    if manifest.get("complete") is not True or status not in {
        "complete",
        "partial_deadline",
    }:
        raise ValueError("evaluation manifest is not a valid final publication")
    sources = manifest.get("source_sha256")
    rates = manifest.get("rates")
    if not isinstance(sources, dict) or not isinstance(rates, dict):
        raise TypeError("evaluation provenance or rate table is malformed")
    results: list[dict[str, object]] = []
    for raw_index in sorted(rates, key=int):
        record = rates[raw_index]
        if not isinstance(record, dict):
            raise TypeError("evaluation rate record must be an object")
        summary_path = _safe_artifact_path(
            evaluation,
            record.get("summary_path"),
            label="evaluation summary path",
        )
        verify_sha256(
            summary_path,
            str(record["summary_sha256"]),
            label=f"evaluation rate {raw_index} summary",
        )
        summary = json.loads(summary_path.read_text())
        if summary.get("source_sha256") != sources:
            raise ValueError("evaluation rate source provenance mismatch")
        for field in ("error_rate", "shots", "status", "stop_reason"):
            if summary.get(field) != record.get(field):
                raise ValueError(f"evaluation rate summary has mismatched {field}")
        results.append(summary)
    return str(status), results


def _verify_scientific_chain(
    campaign: Path,
    *,
    config_path: Path,
    code_path: Path,
    git_commit: str,
) -> None:
    """Cross-check every scientific role against the actual campaign inputs."""
    CampaignConfig.from_json(config_path)
    load_campaign_code(code_path)
    code_manifest_path = code_path / "code.json"
    config_sha256 = sha256_file(config_path)
    code_sha256 = sha256_file(code_manifest_path)
    selection_path = _safe_artifact_path(campaign, "pilot/selection.json", label="pilot selection")
    selection = json.loads(selection_path.read_text())
    if selection.get("source_sha256") != {
        "code_manifest": code_sha256,
        "config": config_sha256,
    }:
        raise ValueError("pilot selection provenance does not match config and code")
    rates = tuple(float(rate) for rate in selection.get("selected_noise_points", ()))
    if not rates:
        raise ValueError("pilot selection must contain noise points")

    shards = {
        role: load_verified_shards(
            campaign / "shards" / role,
            role=role,
            config_path=config_path,
            code_manifest_path=code_manifest_path,
        )
        for role in ("train", "calibration", "test")
    }
    role_seeds = {
        role: {shard.seed for shard in shard_set.shards} for role, shard_set in shards.items()
    }
    if any(
        role_seeds[left] & role_seeds[right]
        for left, right in (("train", "calibration"), ("train", "test"), ("calibration", "test"))
    ):
        raise ValueError("training, calibration, and test seeds are not disjoint")
    for role in ("train", "calibration", "test"):
        rate_map = {shard.rate_index: shard.error_rate for shard in shards[role].shards}
        if (
            set(rate_map) != set(range(len(rates)))
            or tuple(rate_map[index] for index in range(len(rates))) != rates
        ):
            raise ValueError(f"{role} shard rates do not match pilot selection")

    model_path = _safe_artifact_path(campaign, "model/model.json", label="model manifest")
    model = json.loads(model_path.read_text())
    expected_model_sources = {
        "code_manifest": code_sha256,
        "config": config_sha256,
        "train_manifest": shards["train"].manifest_sha256,
        "train_shard_manifests": shards["train"].shard_manifest_sha256,
    }
    if (
        model.get("git_commit") != git_commit
        or model.get("source_sha256") != expected_model_sources
    ):
        raise ValueError("model provenance does not match train shards, config, code, and Git")

    grid_path = _safe_artifact_path(campaign, "calibration/grid.json", label="calibration grid")
    selected_path = _safe_artifact_path(
        campaign, "calibration/selected.json", label="calibration selection"
    )
    selected = json.loads(selected_path.read_text())
    expected_calibration_sources = {
        "calibration_manifest": shards["calibration"].manifest_sha256,
        "calibration_shard_manifests": shards["calibration"].shard_manifest_sha256,
        "code_manifest": code_sha256,
        "config": config_sha256,
        "grid": sha256_file(grid_path),
        "model_manifest": sha256_file(model_path),
    }
    if (
        selected.get("source_role") != "calibration"
        or selected.get("source_sha256") != expected_calibration_sources
    ):
        raise ValueError("calibration provenance does not match calibration-only inputs")

    evaluation_path = _safe_artifact_path(
        campaign, "evaluation/manifest.json", label="evaluation manifest"
    )
    evaluation = json.loads(evaluation_path.read_text())
    expected_evaluation_sources = {
        "calibration_grid": sha256_file(grid_path),
        "calibration_manifest": shards["calibration"].manifest_sha256,
        "calibration_selected": sha256_file(selected_path),
        "calibration_shard_manifests": shards["calibration"].shard_manifest_sha256,
        "code_manifest": code_sha256,
        "config": config_sha256,
        "model_manifest": sha256_file(model_path),
        "selection": sha256_file(selection_path),
        "test_manifest": shards["test"].manifest_sha256,
        "test_shard_manifests": shards["test"].shard_manifest_sha256,
    }
    if evaluation.get("source_sha256") != expected_evaluation_sources:
        raise ValueError("held-out evaluation provenance does not match actual test artifacts")
    evaluation_rates = evaluation.get("rates")
    if (
        not isinstance(evaluation_rates, dict)
        or set(evaluation_rates) != {str(index) for index in range(len(rates))}
        or tuple(float(evaluation_rates[str(index)]["error_rate"]) for index in range(len(rates)))
        != rates
    ):
        raise ValueError("evaluation rates do not match the pilot-selected test rates")


def _summary_markdown(results: dict[str, object]) -> str:
    state = results["completion_state"]
    scope = results.get("scientific_scope", {})
    claims_permitted = (
        scope.get("scientific_claims_permitted", True) if isinstance(scope, dict) else True
    )
    pilot = results.get("pilot")
    model = results.get("model")
    calibration = results.get("calibration")
    selected_rates = pilot.get("selected_noise_points", []) if isinstance(pilot, dict) else []
    model_manifest = model.get("manifest_sha256") if isinstance(model, dict) else None
    model_sha256 = model.get("model_sha256") if isinstance(model, dict) else None
    calibration_grid = calibration.get("grid_sha256") if isinstance(calibration, dict) else None
    calibration_selected = (
        calibration.get("selected_sha256") if isinstance(calibration, dict) else None
    )
    reasons = results.get("early_stop_reasons", [])
    if not isinstance(reasons, list):
        reasons = []
    lines = [
        "# Accuracy campaign results",
        "",
        f"Completion state: **{state}**.",
        "",
    ]
    if not claims_permitted:
        lines.extend(
            [
                "**NON-SCIENTIFIC REDUCED CAMPAIGN:** execution coverage only. ",
                "These measurements must not be reported as scientific results.",
                "",
            ]
        )
    lines.extend(
        [
            "## Scientific scope",
            "",
            (
                "These results cover the `lp(3,7)_16` code at `ell=45` under independent-Z "
                "code-capacity noise with perfect syndrome measurements. Timing is diagnostic only."
            ),
            "",
            "## Reproducibility and provenance",
            "",
            f"- Git commit: `{results.get('git_commit', 'unavailable')}`",
            f"- Pilot-selected noise rates: `{selected_rates}`",
            f"- Model manifest SHA-256: `{model_manifest or 'unavailable'}`",
            f"- Model SHA-256: `{model_sha256 or 'unavailable'}`",
            f"- Calibration grid SHA-256: `{calibration_grid or 'unavailable'}`",
            f"- Calibration selection SHA-256: `{calibration_selected or 'unavailable'}`",
            f"- Early-stop reasons: `{reasons or ['none']}`",
            "",
            "## Held-out test results",
            "",
        ]
    )
    held_out = results["held_out_test_results"]
    if not isinstance(held_out, list) or not held_out:
        lines.append("No verified held-out evaluation was available when this summary was written.")
    else:
        lines.extend(
            [
                "| p | shots | state | stop reason | paired comparison status |",
                "|---:|---:|---|---|---|",
            ]
        )
        for row in held_out:
            comparison_status = row.get("comparison_status", {})
            statuses = (
                "; ".join(
                    f"{method}: {comparison_status.get(method, 'unavailable')}"
                    for method in ("soft_prior", "residual")
                )
                if isinstance(comparison_status, dict)
                else "unavailable"
            )
            lines.append(
                f"| {row.get('error_rate')} | {row.get('shots')} | {row.get('status')} | "
                f"{row.get('stop_reason')} | {statuses} |"
            )
        for row in held_out:
            lines.extend(
                [
                    "",
                    f"### p = {row.get('error_rate')}",
                    "",
                    (
                        "| Decoder | Shots | Failures | Block-error rate | "
                        "95% Wilson interval | Syndrome-valid rate |"
                    ),
                    "|---|---:|---:|---:|---|---:|",
                ]
            )
            decoders = row.get("decoders", {})
            if isinstance(decoders, dict):
                for decoder in ("baseline", "soft_prior", "residual"):
                    metrics = decoders.get(decoder, {})
                    if not isinstance(metrics, dict):
                        continue
                    interval = (
                        f"[{metrics.get('block_error_rate_95ci_low')}, "
                        f"{metrics.get('block_error_rate_95ci_high')}]"
                    )
                    lines.append(
                        f"| {decoder} | {metrics.get('shots', row.get('shots'))} | "
                        f"{metrics.get('block_errors')} | {metrics.get('block_error_rate')} | "
                        f"{interval} | {metrics.get('syndrome_valid_rate')} |"
                    )
            lines.extend(
                [
                    "",
                    "Paired block-error delta (hybrid minus baseline):",
                    "",
                    (
                        "Each hybrid row reports discordant pairs, the exact two-sided McNemar "
                        "p-value, and the 95% Clopper-Pearson interval for hybrid-harm share "
                        "conditional on discordance."
                    ),
                    "",
                    (
                        "| Hybrid | Paired comparison status | Delta | Baseline-only failures | "
                        "Hybrid-only failures | Discordant pairs | exact McNemar two-sided "
                        "p-value | Hybrid-harm share conditional on discordance | 95% "
                        "Clopper-Pearson interval conditional on discordance |"
                    ),
                    "|---|---|---:|---:|---:|---:|---:|---:|---|",
                ]
            )
            paired = row.get("paired", {})
            comparison_status = row.get("comparison_status", {})
            if not isinstance(comparison_status, dict):
                comparison_status = {}
            if isinstance(paired, dict):
                for method in ("soft_prior", "residual"):
                    metrics = paired.get(method, {})
                    if not isinstance(metrics, dict):
                        continue
                    interval = (
                        "["
                        f"{metrics.get('hybrid_harm_share_given_discordance_95ci_low')}, "
                        f"{metrics.get('hybrid_harm_share_given_discordance_95ci_high')}"
                        "]"
                    )
                    lines.append(
                        f"| {method} | {comparison_status.get(method, 'unavailable')} | "
                        f"{metrics.get('block_error_delta')} | "
                        f"{metrics.get('baseline_only_failure')} | "
                        f"{metrics.get('hybrid_only_failure')} | "
                        f"{metrics.get('discordant_pairs')} | "
                        f"{metrics.get('mcnemar_exact_pvalue_two_sided')} | "
                        f"{metrics.get('hybrid_harm_share_given_discordance')} | {interval} |"
                    )
            if "not_fixed_sample" in comparison_status.values():
                lines.extend(
                    [
                        "",
                        (
                            "Paired inference is diagnostic because this result is adaptive, "
                            "incomplete, or both."
                        ),
                    ]
                )
            lines.extend(
                [
                    "",
                    "Timing diagnostics (not an accuracy decision):",
                    "",
                    "| Decoder | Latency basis | p50 seconds | p95 seconds |",
                    "|---|---|---:|---:|",
                ]
            )
            diagnostics = row.get("diagnostics", {})
            if not isinstance(diagnostics, dict):
                diagnostics = {}
            for decoder in ("baseline", "soft_prior", "residual"):
                if decoder == "baseline":
                    decoder_metrics = (
                        decoders.get(decoder, {}) if isinstance(decoders, dict) else {}
                    )
                    latency = (
                        decoder_metrics.get("latency_seconds", {})
                        if isinstance(decoder_metrics, dict)
                        else {}
                    )
                    basis = "BP-LSD"
                else:
                    latency = diagnostics.get(f"{decoder}_end_to_end_latency_seconds", {})
                    basis = "end-to-end"
                if not isinstance(latency, dict):
                    latency = {}
                lines.append(
                    f"| {decoder} | {basis} | {latency.get('p50')} | {latency.get('p95')} |"
                )
    lines.extend(
        [
            "",
            "## Calibration and pilot provenance",
            "",
            (
                "Calibration is not held-out evidence. Pilot rows select operating points only; "
                "calibration rows tune priors only. Neither is included in the held-out table."
            ),
            "",
            "No timing-only winner or speed claim is made by this accuracy campaign.",
            "",
        ]
    )
    return "\n".join(lines)


def write_campaign_summary(
    campaign: Path,
    output: Path,
    *,
    completion_state: CampaignStatus,
    git_commit: str,
    early_stop_reasons: Sequence[str],
    config_path: Path | None = None,
    code_path: Path | None = None,
    campaign_mode: str = "canonical",
    scientific_claims_permitted: bool = True,
) -> dict[str, object]:
    """Write verified JSON/Markdown summaries without mixing scientific roles."""
    if not git_commit:
        raise ValueError("summary Git commit must be non-empty")
    if campaign_mode not in {"canonical", "reduced_non_scientific"}:
        raise ValueError("summary campaign mode is invalid")
    if type(scientific_claims_permitted) is not bool:
        raise TypeError("scientific_claims_permitted must be a boolean")
    if campaign_mode == "reduced_non_scientific" and scientific_claims_permitted:
        raise ValueError("reduced campaign summaries cannot permit scientific claims")
    if output.is_symlink():
        raise ValueError("campaign summary output must not be a symlink")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite campaign summary: {output}")
    output.mkdir(parents=True, exist_ok=True)
    pilot = _verified_pilot(campaign)
    model = _verified_model(campaign)
    calibration = _verified_calibration(campaign)
    evaluation_status, held_out = _verified_evaluation(campaign)
    if evaluation_status is not None:
        if config_path is None or code_path is None:
            raise ValueError(
                "held-out summary provenance verification requires config_path and code_path"
            )
        _verify_scientific_chain(
            campaign,
            config_path=config_path,
            code_path=code_path,
            git_commit=git_commit,
        )
    if completion_state is CampaignStatus.COMPLETE and (
        pilot is None or model is None or calibration is None or evaluation_status != "complete"
    ):
        raise ValueError(
            "complete summary requires verified pilot, model, calibration, and evaluation stages"
        )
    stop_reasons = list(dict.fromkeys(str(reason) for reason in early_stop_reasons if reason))
    for row in held_out:
        reason = row.get("stop_reason")
        if isinstance(reason, str) and reason and reason not in stop_reasons:
            stop_reasons.append(reason)
    results: dict[str, object] = {
        "artifact_completeness": {
            "calibration": calibration is not None,
            "evaluation": evaluation_status is not None,
            "model": model is not None,
            "pilot": pilot is not None,
        },
        "calibration": calibration,
        "completion_state": completion_state.value,
        "early_stop_reasons": stop_reasons,
        "evaluation_status": evaluation_status,
        "git_commit": git_commit,
        "held_out_test_results": held_out,
        "model": model,
        "pilot": pilot,
        "schema_version": 1,
        "scientific_scope": {
            "accuracy_primary": scientific_claims_permitted,
            "campaign_mode": campaign_mode,
            "code": "lp(3,7)_16",
            "ell": 45,
            "noise_model": "independent_Z_code_capacity_perfect_measurements",
            "scientific_claims_permitted": scientific_claims_permitted,
            "speed_claim_permitted": False,
        },
    }
    write_canonical_json(output / "results.json", results)
    (output / "results.md").write_text(_summary_markdown(results))
    return results


def _publish_partial_summary(
    store: ArtifactStore,
    workspace: Path,
    git_commit: str,
    reason: str,
    config_path: Path | None = None,
    code_path: Path | None = None,
    campaign_mode: str = "canonical",
    scientific_claims_permitted: bool = True,
    deadline_monotonic: float | None = None,
) -> str:
    """Publish an immutable, versioned deadline report without completing summary."""
    for index in range(1_000_000):
        prefix = f".partial-summaries/{index:08d}"
        output = workspace / ".partial-summaries" / f"{index:08d}"
        completion_key = f"{prefix}/_COMPLETE.json"
        if store.verify_completion(prefix, deadline_monotonic=deadline_monotonic):
            continue
        if store.exists(
            completion_key,
            deadline_monotonic=deadline_monotonic,
        ):
            raise ValueError(f"partial summary completion is corrupt: {prefix}")
        if output.exists():
            continue
        write_campaign_summary(
            workspace,
            output,
            completion_state=CampaignStatus.PARTIAL_DEADLINE,
            git_commit=git_commit,
            early_stop_reasons=(reason,),
            config_path=config_path,
            code_path=code_path,
            campaign_mode=campaign_mode,
            scientific_claims_permitted=scientific_claims_permitted,
        )
        store.publish_directory(
            output,
            prefix,
            status="partial_deadline",
            deadline_monotonic=deadline_monotonic,
        )
        return prefix
    raise RuntimeError("too many partial deadline summaries")


def _copy_tree_immutably(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, destination)


class _CampaignCommands:
    """Adapters from runner work units to the existing numbered CLIs."""

    def __init__(
        self,
        *,
        config_path: Path,
        code_path: Path,
        workspace: Path,
        command_runner: Callable[[list[str], float | None], None],
        calibration_grid_limit: int | None,
        campaign_mode: str = "canonical",
        monotonic_clock: Callable[[], float] = monotonic,
    ) -> None:
        self.config_path = config_path.resolve()
        self.code_path = code_path.resolve()
        self.workspace = workspace.resolve()
        self.command_runner = command_runner
        self.calibration_grid_limit = calibration_grid_limit
        self.campaign_mode = campaign_mode
        self.monotonic = monotonic_clock
        self.repo = Path(__file__).resolve().parents[3]
        self.experiments = self.repo / "experiments"
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _run(
        self,
        script: str,
        *arguments: object,
        deadline: float | None = None,
    ) -> None:
        path = self.experiments / script
        if not path.is_file():
            raise FileNotFoundError(path)
        command = [sys.executable, str(path), *(str(argument) for argument in arguments)]
        timeout = None if deadline is None else max(0.0, deadline - self.monotonic())
        if timeout == 0.0:
            raise subprocess.TimeoutExpired(command, timeout=0.0)
        self.command_runner(command, timeout)

    def pilot(self, deadline: float | None) -> StageResult:
        output = self.workspace / "pilot"
        if (output / "manifest.json").is_file():
            if _verified_pilot(self.workspace) is None:
                raise ValueError("pilot completion is missing verified selection artifacts")
            load_verified_shards(
                output,
                role="pilot",
                config_path=self.config_path,
                code_manifest_path=self.code_path / "code.json",
            )
            return StageResult.COMPLETE
        self._run(
            "13_pilot_noise_grid.py",
            "--config",
            self.config_path,
            "--code",
            self.code_path,
            "--out",
            output,
            deadline=deadline,
        )
        return StageResult.COMPLETE

    def shards(self, deadline: float | None) -> StageResult:
        root = self.workspace / "shards"
        root.mkdir(exist_ok=True)
        for role in ("train", "calibration", "test"):
            output = root / role
            if (output / "manifest.json").is_file():
                load_verified_shards(
                    output,
                    role=role,
                    config_path=self.config_path,
                    code_manifest_path=self.code_path / "code.json",
                )
                continue
            self._run(
                "14_generate_campaign_shards.py",
                "--config",
                self.config_path,
                "--code",
                self.code_path,
                "--selection",
                self.workspace / "pilot/selection.json",
                "--role",
                role,
                "--out",
                output,
                deadline=deadline,
            )
            return StageResult.COMPLETE if role == "test" else StageResult.CHECKPOINTED
        return StageResult.COMPLETE

    def training(self, deadline: float | None) -> StageResult:
        output = self.workspace / "model"
        arguments: list[object] = [
            "--config",
            self.config_path,
            "--code",
            self.code_path,
            "--train",
            self.workspace / "shards/train",
            "--out",
            output,
        ]
        completed = (output / "model.json").is_file()
        if completed:
            arguments.extend(("--max-epochs-this-run", 1, "--resume"))
        elif not (output / "teacher.json").is_file():
            arguments.extend(("--max-teacher-chunks-this-run", 1, "--prepare-teacher-only"))
        else:
            arguments.extend(("--max-epochs-this-run", 1))
        if output.exists() and not completed:
            arguments.append("--resume")
        self._run("15_train_conditional_fno.py", *arguments, deadline=deadline)
        return (
            StageResult.COMPLETE if (output / "model.json").is_file() else StageResult.CHECKPOINTED
        )

    def calibration(self, deadline: float | None) -> StageResult:
        output = self.workspace / "calibration"
        if (output / "selected.json").is_file():
            if _verified_calibration(self.workspace) is None:
                raise ValueError("calibration completion is missing verified artifacts")
            return StageResult.COMPLETE
        _copy_tree_immutably(self.workspace / "shards/calibration", output)
        arguments: list[object] = [
            "--config",
            self.config_path,
            "--code",
            self.code_path,
            "--calibration",
            output,
            "--model",
            self.workspace / "model",
            "--campaign-mode",
            self.campaign_mode,
            "--out",
            output,
        ]
        if self.calibration_grid_limit is not None:
            arguments.extend(("--grid-limit", self.calibration_grid_limit))
        arguments.extend(("--max-work-units-this-run", 1))
        if (output / "progress.json").is_file():
            arguments.append("--resume")
        self._run("16_calibrate_hybrid_priors.py", *arguments, deadline=deadline)
        return (
            StageResult.COMPLETE
            if (output / "selected.json").is_file()
            else StageResult.CHECKPOINTED
        )

    def evaluation(self, deadline: float | None) -> StageResult:
        output = self.workspace / "evaluation"
        manifest_path = output / "manifest.json"
        if manifest_path.is_file() and json.loads(manifest_path.read_text()).get("status") not in {
            "complete",
            "partial_deadline",
        }:
            raise ValueError("evaluation completion manifest has an invalid status")
        arguments: list[object] = [
            "--config",
            self.config_path,
            "--code",
            self.code_path,
            "--selection",
            self.workspace / "pilot/selection.json",
            "--test",
            self.workspace / "shards/test",
            "--model",
            self.workspace / "model",
            "--calibration",
            self.workspace / "calibration",
            "--campaign-mode",
            self.campaign_mode,
            "--max-batches-this-run",
            1,
            "--out",
            output,
        ]
        if output.exists():
            arguments.append("--resume")
        if deadline is not None:
            arguments.extend(("--deadline-monotonic", deadline))
        self._run("17_evaluate_hybrid_decoders.py", *arguments, deadline=deadline)
        if manifest_path.is_file():
            status = json.loads(manifest_path.read_text()).get("status")
            if status == "complete":
                return StageResult.COMPLETE
            if status == "partial_deadline":
                return StageResult.PARTIAL_DEADLINE
        return StageResult.CHECKPOINTED

    def finalize_evaluation(self, timeout: float) -> bool:
        """Ask the evaluator to publish verified partial results from saved progress."""
        output = self.workspace / "evaluation"
        manifest_path = output / "manifest.json"
        if manifest_path.is_file() or not (output / "progress.json").is_file():
            return False
        arguments: list[object] = [
            "--config",
            self.config_path,
            "--code",
            self.code_path,
            "--selection",
            self.workspace / "pilot/selection.json",
            "--test",
            self.workspace / "shards/test",
            "--model",
            self.workspace / "model",
            "--calibration",
            self.workspace / "calibration",
            "--campaign-mode",
            self.campaign_mode,
            "--deadline-monotonic",
            0,
            "--resume",
            "--out",
            output,
        ]
        path = self.experiments / "17_evaluate_hybrid_decoders.py"
        command = [sys.executable, str(path), *(str(argument) for argument in arguments)]
        self.command_runner(command, timeout)
        if not manifest_path.is_file():
            raise ValueError("evaluation deadline finalization did not publish a manifest")
        return True


def _default_command_runner(command: list[str], timeout: float | None) -> None:
    subprocess.run(command, check=True, timeout=timeout)


def _deadline_finalization_timeout(checkpoint_grace_seconds: float) -> float:
    """Reserve two thirds of grace for summary and checkpoint publication."""
    return max(0.0, checkpoint_grace_seconds / 3.0)


def _git_commit(repo: Path) -> str:
    return resolve_git_commit(repo)


def build_campaign_runner(
    *,
    config_path: Path,
    code_path: Path,
    workspace: Path,
    store: ArtifactStore,
    calibration_grid_limit: int | None = None,
    campaign_mode: str = "canonical",
    scientific_claims_permitted: bool = True,
    command_runner: Callable[[list[str], float | None], None] = _default_command_runner,
    monotonic_clock: Callable[[], float] = monotonic,
) -> CampaignRunner:
    """Compose the existing scientific CLIs into the canonical stage order."""
    config = CampaignConfig.from_json(config_path)
    commands = _CampaignCommands(
        config_path=config_path,
        code_path=code_path,
        workspace=workspace,
        command_runner=command_runner,
        calibration_grid_limit=calibration_grid_limit,
        campaign_mode=campaign_mode,
        monotonic_clock=monotonic_clock,
    )
    commit = _git_commit(commands.repo)

    def summary(deadline: float | None) -> StageResult:
        del deadline
        write_campaign_summary(
            commands.workspace,
            commands.workspace / "summary",
            completion_state=CampaignStatus.COMPLETE,
            git_commit=commit,
            early_stop_reasons=(),
            config_path=commands.config_path,
            code_path=commands.code_path,
            campaign_mode=campaign_mode,
            scientific_claims_permitted=scientific_claims_permitted,
        )
        return StageResult.COMPLETE

    def deadline(reason: str, deadline_monotonic: float | None) -> None:
        finalization_timeout = _deadline_finalization_timeout(config.checkpoint_grace_seconds)
        if deadline_monotonic is not None:
            finalization_timeout = min(
                finalization_timeout,
                max(0.0, deadline_monotonic - monotonic_clock()),
            )
        try:
            commands.finalize_evaluation(finalization_timeout)
        except subprocess.TimeoutExpired:
            pass
        _publish_partial_summary(
            store,
            commands.workspace,
            commit,
            reason,
            config_path=commands.config_path,
            code_path=commands.code_path,
            campaign_mode=campaign_mode,
            scientific_claims_permitted=scientific_claims_permitted,
            deadline_monotonic=deadline_monotonic,
        )

    stages = (
        CampaignStage("pilot", commands.workspace / "pilot", commands.pilot),
        CampaignStage("shards", commands.workspace / "shards", commands.shards),
        CampaignStage("training", commands.workspace / "model", commands.training),
        CampaignStage("calibration", commands.workspace / "calibration", commands.calibration),
        CampaignStage("evaluation", commands.workspace / "evaluation", commands.evaluation),
        CampaignStage("summary", commands.workspace / "summary", summary),
    )
    return CampaignRunner(
        store=store,
        stages=stages,
        checkpoint_grace_seconds=config.checkpoint_grace_seconds,
        monotonic=monotonic_clock,
        on_deadline=deadline,
    )


def _argument_or_environment(value: str | None, name: str) -> str:
    result = value or os.environ.get(name)
    if not result:
        raise ValueError(f"provide the argument or set {name}")
    return result


def _input_execution_identity(
    store_location: str,
    config: CampaignConfig,
) -> dict[str, object]:
    if not store_location.startswith("gs://"):
        return {"kind": "local", "store": str(Path(store_location).resolve())}
    environment_names = (
        "CAMPAIGN_BUCKET",
        "CAMPAIGN_CLOUD_JOB",
        "CAMPAIGN_CLOUD_PROJECT",
        "CAMPAIGN_CLOUD_REGION",
        "CAMPAIGN_IMAGE",
        "CAMPAIGN_IMAGE_DIGEST",
        "CAMPAIGN_FINALIZATION_RESERVE_SECONDS",
        "CAMPAIGN_OUTER_TIMEOUT_SECONDS",
        "CAMPAIGN_PREFIX",
        "CAMPAIGN_SERVICE_ACCOUNT",
        "CAMPAIGN_WORK_CUTOFF_SECONDS",
    )
    values = {name: os.environ.get(name) for name in environment_names}
    if not all(values.values()):
        missing = sorted(name for name, value in values.items() if not value)
        raise ValueError(f"cloud campaign execution identity is incomplete: {missing}")
    expected_seconds = {
        "CAMPAIGN_FINALIZATION_RESERVE_SECONDS": config.checkpoint_grace_seconds,
        "CAMPAIGN_OUTER_TIMEOUT_SECONDS": config.cloud_timeout_seconds,
        "CAMPAIGN_WORK_CUTOFF_SECONDS": config.cloud_timeout_seconds
        - config.checkpoint_grace_seconds,
    }
    for name, expected in expected_seconds.items():
        if values[name] != str(expected):
            raise ValueError(f"cloud campaign execution identity has mismatched {name}")
    expected_store = f"gs://{values['CAMPAIGN_BUCKET']}/{values['CAMPAIGN_PREFIX']}"
    if store_location != expected_store:
        raise ValueError("cloud campaign execution identity has mismatched bucket/prefix/store")
    digest = values["CAMPAIGN_IMAGE_DIGEST"]
    image = values["CAMPAIGN_IMAGE"]
    if (
        not isinstance(digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
        or not isinstance(image, str)
        or not image.endswith(f"@{digest}")
    ):
        raise ValueError("cloud campaign execution identity has malformed image digest")
    return {
        "bucket": values["CAMPAIGN_BUCKET"],
        "finalization_reserve_seconds": config.checkpoint_grace_seconds,
        "image": values["CAMPAIGN_IMAGE"],
        "image_digest": values["CAMPAIGN_IMAGE_DIGEST"],
        "job": values["CAMPAIGN_CLOUD_JOB"],
        "kind": "cloud",
        "outer_timeout_seconds": config.cloud_timeout_seconds,
        "prefix": values["CAMPAIGN_PREFIX"],
        "project": values["CAMPAIGN_CLOUD_PROJECT"],
        "region": values["CAMPAIGN_CLOUD_REGION"],
        "service_account": values["CAMPAIGN_SERVICE_ACCOUNT"],
        "store": store_location,
        "work_cutoff_seconds": config.cloud_timeout_seconds
        - config.checkpoint_grace_seconds,
    }


def _status_payload(status: CampaignStatus) -> dict[str, str]:
    """Build terminal status and an injection-safe cloud resume hint when available."""
    payload = {"status": status.value}
    if status is not CampaignStatus.PARTIAL_DEADLINE:
        return payload
    identity = {
        "CAMPAIGN_CLOUD_JOB": os.environ.get("CAMPAIGN_CLOUD_JOB"),
        "CAMPAIGN_CLOUD_PROJECT": os.environ.get("CAMPAIGN_CLOUD_PROJECT"),
        "CAMPAIGN_CLOUD_REGION": os.environ.get("CAMPAIGN_CLOUD_REGION"),
    }
    if not any(identity.values()):
        return payload
    if not all(identity.values()):
        raise ValueError("cloud resume identity is incomplete")
    patterns = {
        "CAMPAIGN_CLOUD_JOB": r"[a-z][a-z0-9-]{0,62}",
        "CAMPAIGN_CLOUD_PROJECT": r"[a-z][a-z0-9-]{4,28}[a-z0-9]",
        "CAMPAIGN_CLOUD_REGION": r"[a-z][a-z0-9-]{0,30}[a-z0-9]",
    }
    for name, pattern in patterns.items():
        value = identity[name]
        if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
            raise ValueError(f"{name} is invalid")
    job = identity["CAMPAIGN_CLOUD_JOB"]
    assert isinstance(job, str)
    campaign_id = job.removeprefix("qldpc-fno-")
    if campaign_id == job or re.fullmatch(r"[a-z][a-z0-9-]{0,30}[a-z0-9]", campaign_id) is None:
        raise ValueError("CAMPAIGN_CLOUD_JOB does not encode the exact campaign ID")
    payload["resume_command"] = (
        f"CLOUD_PROJECT={identity['CAMPAIGN_CLOUD_PROJECT']} "
        f"CLOUD_REGION={identity['CAMPAIGN_CLOUD_REGION']} "
        f"CAMPAIGN_ID={campaign_id} "
        "bash scripts/launch_cloud_campaign.sh --execute --resume"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--canonical-config")
    parser.add_argument("--code")
    parser.add_argument("--git-commit")
    parser.add_argument("--workdir")
    parser.add_argument("--store", "--output", dest="store")
    parser.add_argument("--deadline-monotonic", type=float)
    parser.add_argument("--calibration-grid-limit", type=int)
    parser.add_argument(
        "--campaign-mode",
        choices=("canonical", "reduced_non_scientific"),
        default="canonical",
    )
    parser.add_argument("--fail-on-stage-execution", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stop-after-inputs", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    source_config_path = Path(_argument_or_environment(args.config, "CAMPAIGN_CONFIG"))
    canonical_config_path = Path(
        _argument_or_environment(args.canonical_config, "CAMPAIGN_CANONICAL_CONFIG")
    )
    source_code_path = Path(_argument_or_environment(args.code, "CAMPAIGN_CODE"))
    git_commit = _argument_or_environment(args.git_commit, "CAMPAIGN_GIT_COMMIT")
    workspace = Path(_argument_or_environment(args.workdir, "CAMPAIGN_WORKDIR"))
    store_location = _argument_or_environment(args.store, "CAMPAIGN_STORE")
    source_config = CampaignConfig.from_json(source_config_path)
    deadline = args.deadline_monotonic
    if deadline is None:
        deadline = monotonic() + source_config.cloud_timeout_seconds
    store = open_artifact_store(store_location)
    prepared = prepare_campaign_inputs(
        store,
        workspace,
        CampaignInputRequest(
            canonical_config=canonical_config_path,
            effective_config=source_config_path,
            code=source_code_path,
            git_commit=git_commit,
            campaign_mode=args.campaign_mode,
            calibration_grid_limit=args.calibration_grid_limit,
            execution_identity=_input_execution_identity(store_location, source_config),
        ),
        deadline_monotonic=deadline,
    )
    if args.stop_after_inputs:
        print(json.dumps({"status": "inputs_complete"}, sort_keys=True))
        return
    command_runner = _default_command_runner
    if args.fail_on_stage_execution:

        def command_runner(command: list[str], timeout: float | None) -> None:
            del timeout
            raise RuntimeError(f"completed resume unexpectedly executed stage command: {command}")

    runner = build_campaign_runner(
        config_path=prepared.config,
        code_path=prepared.code,
        workspace=workspace,
        store=store,
        calibration_grid_limit=args.calibration_grid_limit,
        campaign_mode=args.campaign_mode,
        scientific_claims_permitted=args.campaign_mode == "canonical",
        command_runner=command_runner,
    )
    status = runner.run(deadline)
    print(json.dumps(_status_payload(status), sort_keys=True))


if __name__ == "__main__":
    main()
