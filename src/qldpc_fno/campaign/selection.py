"""Strict verification for pilot and predeclared selection publications."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from qldpc_fno.artifacts import sha256_file
from qldpc_fno.campaign.config import CampaignConfig
from qldpc_fno.campaign.shard_io import (
    VerifiedShardSet,
    load_campaign_code,
    load_verified_shards,
)
from qldpc_fno.campaign.shards import run_pilot_grid, select_noise_points
from qldpc_fno.decoders.bplsd import decode_bplsd_batch
from qldpc_fno.metrics.decoding import score_observable_predictions

_PILOT_ROW_FIELDS = {
    "block_error_rate",
    "block_error_rate_95ci_high",
    "block_error_rate_95ci_low",
    "block_errors",
    "converged",
    "convergence_rate",
    "error_rate",
    "exact_observable_match_rate",
    "latency_mean_seconds",
    "latency_seconds",
    "rate_index",
    "seeds",
    "shots",
    "syndrome_valid",
    "syndrome_valid_rate",
}
_SELECTION_VERIFICATION_FIELDS = {
    "code_manifest_sha256",
    "config_sha256",
    "evidence_role",
    "manifest_sha256",
    "schema_version",
    "selected_noise_points",
    "selection_mode",
    "selection_sha256",
    "verification_algorithm",
}
_SELECTION_VERIFICATION_ALGORITHM = "selection-publication-semantic-v1"


@dataclass(frozen=True, slots=True)
class VerifiedSelectionPublication:
    """Identity and rates recovered from a verified selection publication."""

    evidence_role: str
    manifest_sha256: str
    rates: tuple[float, ...]
    selection_mode: str
    selection_sha256: str


def _type_strict_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(  # type: ignore[arg-type]
            _type_strict_equal(actual[key], value)  # type: ignore[index]
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(  # type: ignore[arg-type]
            _type_strict_equal(item, expected_item)
            for item, expected_item in zip(actual, expected, strict=True)  # type: ignore[arg-type]
        )
    return actual == expected


def _validate_pilot_row_structure(row: dict[str, object]) -> None:
    if set(row) != _PILOT_ROW_FIELDS:
        raise ValueError("pilot row schema does not match the generated publication")
    integer_fields = ("block_errors", "converged", "rate_index", "shots", "syndrome_valid")
    if any(type(row[field]) is not int for field in integer_fields):
        raise ValueError("pilot row deterministic fields have invalid types")
    shots = int(row["shots"])
    if shots <= 0 or any(not 0 <= int(row[field]) <= shots for field in ("block_errors", "converged", "syndrome_valid")):
        raise ValueError("pilot row deterministic counts are invalid")
    float_fields = (
        "block_error_rate",
        "block_error_rate_95ci_high",
        "block_error_rate_95ci_low",
        "convergence_rate",
        "error_rate",
        "exact_observable_match_rate",
        "latency_mean_seconds",
        "latency_seconds",
        "syndrome_valid_rate",
    )
    if any(type(row[field]) is not float or not math.isfinite(row[field]) for field in float_fields):
        raise ValueError("pilot row deterministic fields have invalid numeric values")
    if not isinstance(row["seeds"], list) or any(type(seed) is not int for seed in row["seeds"]):
        raise ValueError("pilot row seeds are malformed")
    latency = float(row["latency_seconds"])
    latency_mean = float(row["latency_mean_seconds"])
    if (
        latency < 0.0
        or latency_mean < 0.0
        or not math.isclose(
            latency_mean,
            latency / shots,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("pilot row timing fields are invalid or internally inconsistent")


def verify_shard_selection_provenance(
    shards: VerifiedShardSet,
    *,
    selection: VerifiedSelectionPublication,
) -> None:
    """Bind a role's exact rate coordinates and manifests to one selection."""
    rate_map: dict[int, float] = {}
    expected_source_fields = {
        "code_manifest",
        "config",
        "dem",
        "pilot_manifest",
        "selection",
    }
    for shard in shards.shards:
        previous = rate_map.setdefault(shard.rate_index, shard.error_rate)
        if previous != shard.error_rate:
            raise ValueError(f"{shards.role} shards have inconsistent rate coordinates")
        if shard.manifest_path.is_symlink():
            raise ValueError(f"{shards.role} shard manifest must not be a symlink")
        manifest_payload = shard.manifest_path.read_bytes()
        if hashlib.sha256(manifest_payload).hexdigest() != shard.manifest_sha256:
            raise ValueError(f"{shards.role} shard manifest changed after verification")
        manifest = json.loads(manifest_payload)
        sources = manifest.get("source_sha256")
        if not isinstance(sources, dict) or set(sources) != expected_source_fields:
            raise ValueError(f"{shards.role} shard source provenance fields are incomplete")
        if sources.get("pilot_manifest") != selection.manifest_sha256:
            raise ValueError(f"{shards.role} shard pilot_manifest provenance mismatch")
        if sources.get("selection") != selection.selection_sha256:
            raise ValueError(f"{shards.role} shard selection provenance mismatch")
    if tuple(rate_map[index] for index in sorted(rate_map)) != selection.rates:
        raise ValueError(f"{shards.role} shard rates do not match the noise-point selection")


def _verify_selection_publication(
    selection_path: Path,
    *,
    config_path: Path,
    code_manifest_path: Path,
    replay_pilot: bool,
    expected_manifest_sha256: str | None = None,
    receipt_selection_sha256: str | None = None,
) -> VerifiedSelectionPublication:
    """Verify a mode-aware stage-13 selection publication and its provenance."""
    config = CampaignConfig.from_json(config_path)
    root = selection_path.parent
    manifest_path = root / "manifest.json"
    if root.is_symlink() or selection_path.is_symlink() or manifest_path.is_symlink():
        raise ValueError("selection publication must not traverse symlinks")
    manifest_payload = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise ValueError("selection manifest SHA-256 disagrees with verification receipt")
    manifest = json.loads(manifest_payload)
    expected_manifest_fields = {
        "complete",
        "role",
        "selection_sha256",
        "shards",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_fields:
        raise ValueError("selection publication manifest schema is malformed")
    if manifest.get("complete") is not True or manifest.get("role") != "pilot":
        raise ValueError("selection does not belong to a completed pilot publication")
    declared_shards = manifest.get("shards")
    if not isinstance(declared_shards, dict) or any(
        not isinstance(relative, str) or not isinstance(digest, str)
        for relative, digest in declared_shards.items()
    ):
        raise ValueError("selection publication shard table is malformed")
    manifest_selection_sha256 = manifest.get("selection_sha256")
    if not isinstance(manifest_selection_sha256, str):
        raise TypeError("selection publication SHA-256 must be a string")
    selection_payload = selection_path.read_bytes()
    selection_sha256 = hashlib.sha256(selection_payload).hexdigest()
    if receipt_selection_sha256 is not None and selection_sha256 != receipt_selection_sha256:
        raise ValueError("selection SHA-256 disagrees with verification receipt")
    if selection_sha256 != manifest_selection_sha256:
        raise ValueError(
            "selection SHA-256 mismatch: "
            f"expected {manifest_selection_sha256}, found {selection_sha256}"
        )
    selection = json.loads(selection_payload)
    expected_selection_fields = {
        "evidence_role",
        "pilot_rows",
        "selected_noise_points",
        "selection_mode",
        "source_sha256",
    }
    if not isinstance(selection, dict) or set(selection) != expected_selection_fields:
        raise ValueError("selection publication schema is malformed")
    sources = selection.get("source_sha256")
    if not isinstance(sources, dict) or set(sources) != {"code_manifest", "config"}:
        raise TypeError("selection is missing exact source SHA-256 provenance")
    if sources.get("config") != sha256_file(config_path):
        raise ValueError("campaign configuration SHA-256 mismatch in selection provenance")
    if sources.get("code_manifest") != sha256_file(code_manifest_path):
        raise ValueError("code manifest SHA-256 mismatch in selection provenance")
    if selection.get("selection_mode") != config.selection_mode:
        raise ValueError("selection mode does not match campaign configuration")

    evidence_roles = {
        "fixed": "predeclared_selection_not_evidence",
        "pilot": "selection_only_not_held_out",
    }
    evidence_role = selection.get("evidence_role")
    if evidence_role != evidence_roles[config.selection_mode]:
        raise ValueError("selection evidence role does not match campaign configuration")
    pilot_rows = selection.get("pilot_rows")
    if not isinstance(pilot_rows, list):
        raise TypeError("selection pilot_rows must be a list")
    if config.selection_mode == "fixed" and pilot_rows:
        raise ValueError("fixed selection publication requires empty pilot_rows")
    if config.selection_mode == "fixed" and declared_shards:
        raise ValueError("fixed selection publication requires an empty shard table")
    if config.selection_mode == "fixed" and any(root.glob("rate-*")):
        raise ValueError("fixed selection publication must not contain rate artifacts")
    if config.selection_mode == "pilot":
        if not pilot_rows or any(not isinstance(row, dict) for row in pilot_rows):
            raise ValueError("pilot selection requires non-empty object-valued pilot_rows")
        for row in pilot_rows:
            _validate_pilot_row_structure(row)

    raw_rates = selection.get("selected_noise_points")
    if not isinstance(raw_rates, list) or not raw_rates:
        raise ValueError("selection must contain non-empty selected_noise_points")
    if any(
        isinstance(rate, bool) or type(rate) not in (int, float) or not math.isfinite(rate)
        for rate in raw_rates
    ):
        raise ValueError("selected noise points must be finite numbers")
    rates = tuple(float(rate) for rate in raw_rates)
    if any(not 0.0 < rate < 0.5 for rate in rates) or any(
        left >= right for left, right in pairwise(rates)
    ):
        raise ValueError("selected noise points must be strictly increasing probabilities")
    if config.selection_mode == "fixed" and rates != config.noise_grid:
        raise ValueError("fixed selection rates do not match configured noise_grid")

    if config.selection_mode == "pilot":
        pilot_shards = load_verified_shards(
            root,
            role="pilot",
            config_path=config_path,
            code_manifest_path=code_manifest_path,
        )
        if pilot_shards.manifest_sha256 != manifest_sha256:
            raise ValueError("pilot completion manifest changed during verification")
        shards_by_rate = {
            rate_index: tuple(
                sorted(
                    (
                        shard
                        for shard in pilot_shards.shards
                        if shard.rate_index == rate_index
                    ),
                    key=lambda shard: shard.shard_index,
                )
            )
            for rate_index in {shard.rate_index for shard in pilot_shards.shards}
        }
        if set(shards_by_rate) != set(range(len(pilot_rows))):
            raise ValueError("pilot rows do not match verified pilot shard rate coordinates")
        if replay_pilot:
            _, hx, _, logical_x = load_campaign_code(code_manifest_path.parent)
        structural_block_errors: list[int] = []
        for rate_index, row in enumerate(pilot_rows):
            rate_shards = shards_by_rate[rate_index]
            expected_shots = sum(shard.stop - shard.start for shard in rate_shards)
            expected_seeds = [shard.seed for shard in rate_shards]
            if type(row.get("rate_index")) is not int or row["rate_index"] != rate_index:
                raise ValueError("pilot row rate_index does not match verified pilot shards")
            if (
                isinstance(row.get("error_rate"), bool)
                or type(row.get("error_rate")) not in (int, float)
                or row["error_rate"] != rate_shards[0].error_rate
            ):
                raise ValueError("pilot row error_rate does not match verified pilot shards")
            if type(row.get("shots")) is not int or row["shots"] != expected_shots:
                raise ValueError("pilot row shots do not match verified pilot shards")
            if expected_shots != config.pilot_shots_per_point:
                raise ValueError("pilot shard shots do not match configured pilot_shots_per_point")
            if row.get("seeds") != expected_seeds:
                raise ValueError("pilot row seeds do not match verified pilot shards")
            block_errors = row.get("block_errors")
            if (
                type(block_errors) is not int
                or block_errors < 0
                or block_errors > expected_shots
            ):
                raise ValueError("pilot row block_errors must be between zero and shots")
            structural_block_errors.append(block_errors)
            if not replay_pilot:
                continue
            indices = pilot_shards.indices_for_rate(rate_index)
            result = decode_bplsd_batch(
                hx,
                pilot_shards.read("dets.b8", indices),
                logical_x,
                error_rate=rate_shards[0].error_rate,
            )
            score = score_observable_predictions(
                pilot_shards.read("obs_actual.b8", indices),
                result.predicted_observables,
                syndrome_valid=result.syndrome_valid,
            )
            expected_deterministic = {
                **score,
                "converged": int(np.count_nonzero(result.converged)),
                "convergence_rate": float(np.mean(result.converged)),
                "error_rate": rate_shards[0].error_rate,
                "rate_index": rate_index,
                "seeds": expected_seeds,
                "syndrome_valid": int(np.count_nonzero(result.syndrome_valid)),
                "syndrome_valid_rate": float(np.mean(result.syndrome_valid)),
            }
            verified_block_error_count = int(score["block_errors"])
            if block_errors != verified_block_error_count:
                raise ValueError("pilot row block_errors disagree with verified pilot outcomes")
            if any(
                type(row[field]) is not type(expected)
                or row[field] != expected
                for field, expected in expected_deterministic.items()
            ):
                raise ValueError("pilot row deterministic fields disagree with verified outcomes")

        def verify_grid_rate(rate: float, rate_index: int) -> dict[str, object]:
            if rate_index >= len(pilot_rows) or pilot_rows[rate_index]["error_rate"] != rate:
                raise ValueError("pilot rows do not follow the configured pilot rate trajectory")
            return {"block_errors": structural_block_errors[rate_index]}

        expected_pilot_rows = run_pilot_grid(config.noise_grid, verify_grid_rate)
        if len(expected_pilot_rows) != len(pilot_rows):
            raise ValueError("pilot rows do not follow the configured pilot rate trajectory")
        if tuple(select_noise_points(pilot_rows)) != rates:
            raise ValueError("selected noise points disagree with pilot rows")

    return VerifiedSelectionPublication(
        evidence_role=str(evidence_role),
        manifest_sha256=manifest_sha256,
        rates=rates,
        selection_mode=config.selection_mode,
        selection_sha256=selection_sha256,
    )


def verify_selection_publication(
    selection_path: Path,
    *,
    config_path: Path,
    code_manifest_path: Path,
) -> VerifiedSelectionPublication:
    """Fully verify a selection publication, including pilot outcome replay."""
    return _verify_selection_publication(
        selection_path,
        config_path=config_path,
        code_manifest_path=code_manifest_path,
        replay_pilot=True,
    )


def selection_verification_receipt(
    selection: VerifiedSelectionPublication,
    *,
    config_path: Path,
    code_manifest_path: Path,
) -> dict[str, object]:
    """Build the durable identity receipt for a semantically verified selection."""
    return {
        "code_manifest_sha256": sha256_file(code_manifest_path),
        "config_sha256": sha256_file(config_path),
        "evidence_role": selection.evidence_role,
        "manifest_sha256": selection.manifest_sha256,
        "schema_version": 1,
        "selected_noise_points": list(selection.rates),
        "selection_mode": selection.selection_mode,
        "selection_sha256": selection.selection_sha256,
        "verification_algorithm": _SELECTION_VERIFICATION_ALGORITHM,
    }


def verify_selection_verification_receipt(
    receipt_path: Path,
    *,
    selection_path: Path,
    config_path: Path,
    code_manifest_path: Path,
) -> tuple[VerifiedSelectionPublication, str]:
    """Hash-verify a selection already replayed at this durable trust boundary."""
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("selection verification receipt must be a regular file")
    payload_bytes = receipt_path.read_bytes()
    receipt_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    receipt = json.loads(payload_bytes)
    if not isinstance(receipt, dict) or set(receipt) != _SELECTION_VERIFICATION_FIELDS:
        raise ValueError("selection verification receipt schema is malformed")
    if type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1:
        raise ValueError("selection verification receipt schema version is unsupported")
    manifest_sha256 = receipt.get("manifest_sha256")
    selection_sha256 = receipt.get("selection_sha256")
    if not isinstance(manifest_sha256, str) or not isinstance(selection_sha256, str):
        raise TypeError("selection verification receipt digests must be strings")
    verified = _verify_selection_publication(
        selection_path,
        config_path=config_path,
        code_manifest_path=code_manifest_path,
        replay_pilot=False,
        expected_manifest_sha256=manifest_sha256,
        receipt_selection_sha256=selection_sha256,
    )
    expected = selection_verification_receipt(
        verified,
        config_path=config_path,
        code_manifest_path=code_manifest_path,
    )
    if not _type_strict_equal(receipt, expected):
        raise ValueError("selection verification receipt disagrees with verified provenance")
    return verified, receipt_sha256
