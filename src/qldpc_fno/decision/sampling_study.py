"""Budgeted sampling comparison against exact logical-class mass."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.decision.exact_css import canonical_steane_z_problem, exact_posterior
from qldpc_fno.decision.sampling import (
    ConditionalSamples,
    logical_mass_metrics,
    metropolis_affine_sample,
    metropolis_affine_sample_for_proposals,
    rejection_sample,
    rejection_sample_for_proposals,
)

_METHODS = ("conditional_rejection", "affine_metropolis")
_SOURCE_PATHS = (
    Path(__file__),
    Path(__file__).with_name("sampling.py"),
    Path(__file__).with_name("exact_css.py"),
)


def _load_configs(
    decision_path: Path, sampling_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    decision = json.loads(decision_path.read_text())
    sampling = json.loads(sampling_path.read_text())
    if decision.get("schema_version") != 1 or sampling.get("schema_version") != 1:
        raise ValueError("decision and sampling schema versions must both be 1")
    domain = str(sampling["seed_domain"])
    expected_seed = int.from_bytes(hashlib.sha256(domain.encode()).digest()[:8], "big")
    if int(sampling["campaign_seed"]) != expected_seed:
        raise ValueError("campaign_seed must be the SHA-256 derivation of seed_domain")
    budgets = sampling.get("sample_budgets")
    if (
        not isinstance(budgets, list)
        or not budgets
        or any(type(value) is not int or value <= 0 for value in budgets)
        or budgets != sorted(set(budgets))
    ):
        raise ValueError("sample_budgets must be unique increasing positive integers")
    return decision, sampling


def _seed(
    campaign_seed: int, *, method: str, channel: str, syndrome_index: int, budget: int
) -> int:
    identity = (
        f"qldpc-fno/mass-sampling/stream/v1:{campaign_seed}:{method}:"
        f"{channel}:{syndrome_index}:{budget}"
    )
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")


def _sample_row(
    *,
    channel_id: str,
    syndrome: np.ndarray,
    syndrome_probability: float,
    exact,
    budget_axis: str,
    requested_budget: int,
    seed: int,
    samples: ConditionalSamples,
) -> dict[str, object]:
    problem = canonical_steane_z_problem()
    metrics = logical_mass_metrics(problem, samples.samples, exact)
    return {
        "true_channel": channel_id,
        "syndrome": syndrome.tolist(),
        "syndrome_probability": syndrome_probability,
        "method": samples.method,
        "budget_axis": budget_axis,
        "requested_budget": requested_budget,
        "sample_budget": int(samples.samples.shape[0]),
        "proposal_budget": (requested_budget if budget_axis == "complete_proposals" else None),
        "seed": seed,
        "proposal_count": samples.proposal_count,
        "accepted_transition_count": samples.accepted_transition_count,
        "operation_counts": samples.operation_counts,
        "estimated_logical_class_probabilities": (
            metrics.estimated_logical_class_probabilities.tolist()
        ),
        "exact_logical_class_probabilities": exact.logical_class_probabilities.tolist(),
        "total_variation": metrics.total_variation,
        "worst_class_absolute_error": metrics.worst_class_absolute_error,
        "top_class_correct": metrics.top_class_correct,
        "physical_state_coverage": metrics.physical_state_coverage,
        "effective_sample_size": metrics.effective_sample_size,
    }


def _summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = tuple(
        dict.fromkeys(
            (
                str(row["budget_axis"]),
                str(row["method"]),
                None if row["requested_budget"] is None else int(row["requested_budget"]),
            )
            for row in rows
        )
    )
    summaries: list[dict[str, object]] = []
    for budget_axis, method, requested_budget in groups:
        selected = [
            row
            for row in rows
            if row["budget_axis"] == budget_axis
            and row["method"] == method
            and row["requested_budget"] == requested_budget
        ]
        ess_values = [
            float(row["effective_sample_size"])
            for row in selected
            if row["effective_sample_size"] is not None
        ]
        operation_keys = tuple(selected[0]["operation_counts"])
        summaries.append(
            {
                "method": method,
                "budget_axis": budget_axis,
                "requested_budget": requested_budget,
                "state_count": len(selected),
                "mean_total_variation": float(
                    np.mean([float(row["total_variation"]) for row in selected])
                ),
                "maximum_total_variation": max(float(row["total_variation"]) for row in selected),
                "top_class_accuracy": float(
                    np.mean([bool(row["top_class_correct"]) for row in selected])
                ),
                "mean_physical_state_coverage": float(
                    np.mean([float(row["physical_state_coverage"]) for row in selected])
                ),
                "mean_operation_counts": {
                    key: float(np.mean([int(row["operation_counts"][key]) for row in selected]))
                    for key in operation_keys
                },
                "ess_available_state_count": len(ess_values),
                "ess_available_state_fraction": len(ess_values) / len(selected),
                "mean_available_effective_sample_size": (
                    None if not ess_values else float(np.mean(ess_values))
                ),
            }
        )
    return summaries


def run_mass_sampling_study(
    decision_config_path: Path,
    sampling_config_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Evaluate exact, rejection, and strong affine-Metropolis mass estimates."""
    decision, sampling = _load_configs(decision_config_path, sampling_config_path)
    problem = canonical_steane_z_problem()
    rows: list[dict[str, object]] = []
    campaign_seed = int(sampling["campaign_seed"])
    burn_in = int(sampling["metropolis_burn_in"])
    thinning = int(sampling["metropolis_thinning"])

    for channel in decision["true_channels"]:
        channel_id = str(channel["id"])
        probabilities = np.asarray(channel["probabilities"], dtype=np.float64)
        for syndrome_index in range(1 << problem.hx.shape[0]):
            syndrome = np.array(
                [(syndrome_index >> shift) & 1 for shift in range(problem.hx.shape[0])],
                dtype=np.uint8,
            )
            exact = exact_posterior(problem, syndrome, probabilities)
            syndrome_probability = float(exact.unnormalized_logical_class_masses.sum())
            rows.append(
                {
                    "true_channel": channel_id,
                    "syndrome": syndrome.tolist(),
                    "syndrome_probability": syndrome_probability,
                    "method": "exact_enumeration",
                    "budget_axis": "exact",
                    "requested_budget": None,
                    "sample_budget": None,
                    "proposal_budget": None,
                    "seed": None,
                    "proposal_count": len(exact.compatible_errors),
                    "accepted_transition_count": None,
                    "operation_counts": {
                        "setup_candidate_error_materializations": 1 << problem.n,
                        "setup_syndrome_evaluations": 1 << problem.n,
                        "setup_affine_terminal_materializations": len(exact.compatible_errors),
                        "channel_error_draws": 0,
                        "sampling_syndrome_evaluations": 0,
                        "target_log_weight_evaluations": len(exact.compatible_errors),
                        "affine_transition_proposals": 0,
                        "burn_in_transitions": 0,
                        "retained_samples": len(exact.compatible_errors),
                    },
                    "estimated_logical_class_probabilities": (
                        exact.logical_class_probabilities.tolist()
                    ),
                    "exact_logical_class_probabilities": exact.logical_class_probabilities.tolist(),
                    "total_variation": 0.0,
                    "worst_class_absolute_error": 0.0,
                    "top_class_correct": True,
                    "physical_state_coverage": 1.0,
                    "effective_sample_size": None,
                }
            )
            for budget in sampling["sample_budgets"]:
                for method in _METHODS:
                    stream_seed = _seed(
                        campaign_seed,
                        method=method,
                        channel=channel_id,
                        syndrome_index=syndrome_index,
                        budget=int(budget),
                    )
                    if method == "conditional_rejection":
                        sample_result = rejection_sample(
                            problem,
                            syndrome,
                            probabilities,
                            sample_count=int(budget),
                            seed=stream_seed,
                        )
                    else:
                        sample_result = metropolis_affine_sample(
                            problem,
                            syndrome,
                            probabilities,
                            sample_count=int(budget),
                            seed=stream_seed,
                            burn_in=burn_in,
                            thinning=thinning,
                        )
                    rows.append(
                        _sample_row(
                            channel_id=channel_id,
                            syndrome=syndrome,
                            syndrome_probability=syndrome_probability,
                            exact=exact,
                            budget_axis="retained",
                            requested_budget=int(budget),
                            seed=stream_seed,
                            samples=sample_result,
                        )
                    )
            for proposal_budget in sampling["generated_configuration_budgets"]:
                for method in _METHODS:
                    stream_seed = _seed(
                        campaign_seed,
                        method=f"{method}/complete_proposals",
                        channel=channel_id,
                        syndrome_index=syndrome_index,
                        budget=int(proposal_budget),
                    )
                    if method == "conditional_rejection":
                        sample_result = rejection_sample_for_proposals(
                            problem,
                            syndrome,
                            probabilities,
                            proposal_budget=int(proposal_budget),
                            seed=stream_seed,
                        )
                    else:
                        sample_result = metropolis_affine_sample_for_proposals(
                            problem,
                            syndrome,
                            probabilities,
                            proposal_budget=int(proposal_budget),
                            seed=stream_seed,
                            burn_in=burn_in,
                            thinning=thinning,
                        )
                    rows.append(
                        _sample_row(
                            channel_id=channel_id,
                            syndrome=syndrome,
                            syndrome_probability=syndrome_probability,
                            exact=exact,
                            budget_axis="complete_proposals",
                            requested_budget=int(proposal_budget),
                            seed=stream_seed,
                            samples=sample_result,
                        )
                    )

    payload: dict[str, object] = {
        "schema_version": 1,
        "claim_boundary": (
            "Exact and seeded sampling diagnostics on one Steane code under two "
            "declared independent-Z channels; no learned model or latency claim."
        ),
        "provenance": {
            "decision_config_sha256": sha256_file(decision_config_path),
            "sampling_config_sha256": sha256_file(sampling_config_path),
            "source_sha256": {path.name: sha256_file(path) for path in _SOURCE_PATHS},
        },
        "sampling_policy": {
            "rejection": "independent physical-channel proposals conditioned by rejection",
            "metropolis": ("symmetric uniform proposal over all 15 nonzero affine displacements"),
            "metropolis_burn_in": burn_in,
            "metropolis_thinning": thinning,
            "operation_count_boundary": (
                "Setup materialization, channel draws, syndrome evaluations, affine "
                "proposals, and target-weight evaluations are reported separately. "
                "The complete-proposal ladder matches stochastic complete-state proposals "
                "but does not pretend their primitive costs or one-time setup are equal. "
                "No latency ordering is inferred."
            ),
        },
        "rows": rows,
        "summaries": _summaries(rows),
    }
    write_canonical_json(output_dir / "mass_sampling.json", payload)
    return payload
