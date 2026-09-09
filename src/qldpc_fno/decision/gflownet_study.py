"""Held-out trajectory-balance GFlowNet experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.decision.exact_css import canonical_steane_z_problem, exact_posterior
from qldpc_fno.decision.gflownet import (
    AffineTrajectoryEnvironment,
    TrajectoryBalanceGFlowNet,
    induced_terminal_probabilities,
    sample_terminal_errors,
    train_trajectory_balance,
)
from qldpc_fno.decision.sampling import logical_mass_metrics

_ROOT = Path(__file__).parents[3]
_CANONICAL_CONFIG = _ROOT / "configs" / "gflownet_mass.json"
_SOURCE_PATHS = (
    Path(__file__),
    Path(__file__).with_name("gflownet.py"),
    Path(__file__).with_name("exact_css.py"),
    Path(__file__).with_name("sampling.py"),
)


def _load_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("GFlowNet schema_version must be 1")
    domain = str(payload["seed_domain"])
    expected_seeds = [
        int.from_bytes(hashlib.sha256(f"{domain}/train/{index}".encode()).digest()[:8], "big")
        for index in range(len(payload["training_seeds"]))
    ]
    if payload["training_seeds"] != expected_seeds:
        raise ValueError("training_seeds must be consecutive domain-derived seeds")
    expected_sampling_seeds = [
        int.from_bytes(hashlib.sha256(f"{domain}/sample/{index}".encode()).digest()[:8], "big")
        for index in range(len(payload["sampling_replicate_seeds"]))
    ]
    if payload["sampling_replicate_seeds"] != expected_sampling_seeds:
        raise ValueError("sampling_replicate_seeds must be separately domain-derived")
    training_syndrome_list = payload["training_syndrome_indices"]
    heldout_syndrome_list = payload["heldout_syndrome_indices"]
    training_syndromes = set(training_syndrome_list)
    heldout_syndromes = set(heldout_syndrome_list)
    if len(training_syndromes) != len(training_syndrome_list) or len(heldout_syndromes) != len(
        heldout_syndrome_list
    ):
        raise ValueError("syndrome index lists must not contain duplicates")
    if not training_syndromes or not heldout_syndromes or training_syndromes & heldout_syndromes:
        raise ValueError("training and heldout syndrome sets must be nonempty and disjoint")
    if not training_syndromes | heldout_syndromes <= set(range(8)):
        raise ValueError("Steane syndrome indices must lie in [0, 7]")
    field_ids: set[str] = set()
    probability_vectors: set[tuple[float, ...]] = set()
    for field in payload["training_fields"]:
        field_id = str(field["id"])
        probabilities = np.asarray(field["probabilities"], dtype=np.float64)
        if probabilities.shape != (7,) or not np.all((probabilities > 0) & (probabilities < 0.5)):
            raise ValueError("training fields require seven probabilities in (0, 0.5)")
        if field_id in field_ids or tuple(probabilities) in probability_vectors:
            raise ValueError("training field ids and probability vectors must be unique")
        field_ids.add(field_id)
        probability_vectors.add(tuple(probabilities))
    return payload


def _syndrome(index: int) -> np.ndarray:
    return np.array([(index >> shift) & 1 for shift in range(3)], dtype=np.uint8)


def _validate_decision_fields(decision: dict[str, object]) -> None:
    if decision.get("schema_version") != 1 or not isinstance(decision.get("true_channels"), list):
        raise ValueError("decision configuration must contain schema-v1 true channels")
    ids: set[str] = set()
    vectors: set[tuple[float, ...]] = set()
    for field in decision["true_channels"]:
        field_id = str(field["id"])
        probabilities = np.asarray(field["probabilities"], dtype=np.float64)
        vector = tuple(float(value) for value in probabilities)
        if probabilities.shape != (7,) or not np.all((probabilities > 0) & (probabilities < 0.5)):
            raise ValueError("heldout fields require seven probabilities in (0, 0.5)")
        if field_id in ids or vector in vectors:
            raise ValueError("heldout field ids and probability vectors must be unique")
        ids.add(field_id)
        vectors.add(vector)


def _sample_seed(replicate_seed: int, split: str, field_id: str, syndrome: int, budget: int) -> int:
    identity = (
        f"qldpc-fno/gflownet-mass/sample/v1:{replicate_seed}:{split}:{field_id}:{syndrome}:{budget}"
    )
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")


def _unique_top_class_correct(probabilities, *, exact_class: int) -> bool:
    values = np.asarray(probabilities, dtype=np.float64)
    winners = np.flatnonzero(values == values.max())
    return len(winners) == 1 and int(winners[0]) == exact_class


def _environment(field: dict[str, object], syndrome_index: int) -> AffineTrajectoryEnvironment:
    return AffineTrajectoryEnvironment(
        canonical_steane_z_problem(),
        _syndrome(syndrome_index),
        np.asarray(field["probabilities"], dtype=np.float64),
    )


def _evaluation_contexts(
    config: dict[str, object], decision: dict[str, object]
) -> list[tuple[str, dict[str, object], int]]:
    result: list[tuple[str, dict[str, object], int]] = []
    for field in config["training_fields"]:
        for syndrome_index in config["training_syndrome_indices"]:
            result.append(("training_fit", field, int(syndrome_index)))
    for field in config["training_fields"]:
        for syndrome_index in config["heldout_syndrome_indices"]:
            result.append(("heldout_syndrome", field, int(syndrome_index)))
    for field in decision["true_channels"]:
        for syndrome_index in range(8):
            result.append(("heldout_field", field, syndrome_index))
    return result


def _evaluate(
    model: TrajectoryBalanceGFlowNet,
    *,
    training_seed: int,
    split: str,
    field: dict[str, object],
    syndrome_index: int,
    budgets: list[int],
    sampling_replicate_seeds: list[int],
) -> dict[str, object]:
    environment = _environment(field, syndrome_index)
    exact = exact_posterior(environment.problem, environment.syndrome, environment.probabilities)
    induced = induced_terminal_probabilities(model, environment)
    exact_physical = np.exp(environment.log_rewards)
    exact_physical /= exact_physical.sum()
    induced_logical = np.array(
        [induced[environment.logical_classes == label].sum() for label in (0, 1)]
    )
    physical_error = np.abs(induced - exact_physical)
    logical_error = np.abs(induced_logical - exact.logical_class_probabilities)
    sampled: list[dict[str, object]] = []
    for budget in budgets:
        for replicate_index, replicate_seed in enumerate(sampling_replicate_seeds):
            seed = _sample_seed(replicate_seed, split, str(field["id"]), syndrome_index, budget)
            samples = sample_terminal_errors(model, environment, sample_count=budget, seed=seed)
            metrics = logical_mass_metrics(environment.problem, samples, exact)
            sampled.append(
                {
                    "sample_budget": budget,
                    "sampling_replicate_index": replicate_index,
                    "sampling_replicate_seed": replicate_seed,
                    "derived_seed": seed,
                    "estimated_logical_class_probabilities": (
                        metrics.estimated_logical_class_probabilities.tolist()
                    ),
                    "logical_total_variation": metrics.total_variation,
                    "worst_class_absolute_error": metrics.worst_class_absolute_error,
                    "top_class_correct": metrics.top_class_correct,
                    "physical_state_coverage": metrics.physical_state_coverage,
                    "logical_effective_sample_size": metrics.effective_sample_size,
                    "operation_counts": {
                        "generated_valid_terminal_materializations": budget,
                        "terminal_logical_signature_evaluations": budget,
                        "batched_policy_forward_invocations": model.coefficient_width,
                        "terminal_action_logit_evaluations": (budget * model.coefficient_width),
                        "inference_reward_evaluations": 0,
                    },
                }
            )
    context = torch.as_tensor(environment.context[None, :], dtype=torch.float32)
    with torch.no_grad():
        predicted_log_partition = float(model.log_z(context)[0])
    exact_log_partition = float(np.log(np.exp(environment.log_rewards).sum()))
    return {
        "training_seed": training_seed,
        "evaluation_split": split,
        "field_id": field["id"],
        "syndrome_index": syndrome_index,
        "syndrome": environment.syndrome.tolist(),
        "trajectory_multiplicity": 1,
        "exact_logical_class_probabilities": exact.logical_class_probabilities.tolist(),
        "induced_logical_class_probabilities": induced_logical.tolist(),
        "induced_physical_total_variation": float(0.5 * physical_error.sum()),
        "induced_logical_total_variation": float(0.5 * logical_error.sum()),
        "induced_worst_class_absolute_error": float(logical_error.max()),
        "induced_top_class_correct": _unique_top_class_correct(
            induced_logical, exact_class=exact.coset_map_class
        ),
        "predicted_log_partition": predicted_log_partition,
        "exact_log_partition": exact_log_partition,
        "absolute_log_partition_error": abs(predicted_log_partition - exact_log_partition),
        "sampled_metrics": sampled,
    }


def _summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for split in ("training_fit", "heldout_syndrome", "heldout_field"):
        selected = [row for row in rows if row["evaluation_split"] == split]
        summaries.append(
            {
                "evaluation_split": split,
                "row_count": len(selected),
                "training_seed_count": len({row["training_seed"] for row in selected}),
                "mean_induced_physical_total_variation": float(
                    np.mean([row["induced_physical_total_variation"] for row in selected])
                ),
                "maximum_induced_physical_total_variation": max(
                    row["induced_physical_total_variation"] for row in selected
                ),
                "mean_induced_logical_total_variation": float(
                    np.mean([row["induced_logical_total_variation"] for row in selected])
                ),
                "induced_top_class_accuracy": float(
                    np.mean([row["induced_top_class_correct"] for row in selected])
                ),
                "mean_absolute_log_partition_error": float(
                    np.mean([row["absolute_log_partition_error"] for row in selected])
                ),
            }
        )
    return summaries


def _sampled_summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    splits = ("training_fit", "heldout_syndrome", "heldout_field")
    budgets = sorted(
        {int(sample["sample_budget"]) for row in rows for sample in row["sampled_metrics"]}
    )
    for split in splits:
        split_rows = [row for row in rows if row["evaluation_split"] == split]
        for budget in budgets:
            samples = [
                sample
                for row in split_rows
                for sample in row["sampled_metrics"]
                if sample["sample_budget"] == budget
            ]
            ess_values = [
                float(sample["logical_effective_sample_size"])
                for sample in samples
                if sample["logical_effective_sample_size"] is not None
            ]
            unit_summaries: list[dict[str, object]] = []
            unit_keys = sorted(
                {
                    (int(row["training_seed"]), int(sample["sampling_replicate_index"]))
                    for row in split_rows
                    for sample in row["sampled_metrics"]
                    if sample["sample_budget"] == budget
                }
            )
            for training_seed, replicate_index in unit_keys:
                unit_samples = [
                    sample
                    for row in split_rows
                    if row["training_seed"] == training_seed
                    for sample in row["sampled_metrics"]
                    if sample["sample_budget"] == budget
                    and sample["sampling_replicate_index"] == replicate_index
                ]
                unit_summaries.append(
                    {
                        "training_seed": training_seed,
                        "sampling_replicate_index": replicate_index,
                        "context_count": len(unit_samples),
                        "mean_logical_total_variation": float(
                            np.mean([sample["logical_total_variation"] for sample in unit_samples])
                        ),
                        "top_class_accuracy": float(
                            np.mean([sample["top_class_correct"] for sample in unit_samples])
                        ),
                    }
                )
            result.append(
                {
                    "evaluation_split": split,
                    "sample_budget": budget,
                    "sample_row_count": len(samples),
                    "training_seed_count": len({row["training_seed"] for row in split_rows}),
                    "sampling_replicate_count": len(
                        {sample["sampling_replicate_index"] for sample in samples}
                    ),
                    "training_seed_sampling_replicate_unit_count": len(unit_summaries),
                    "unit_summaries": unit_summaries,
                    "mean_of_unit_mean_logical_total_variation": float(
                        np.mean([unit["mean_logical_total_variation"] for unit in unit_summaries])
                    ),
                    "maximum_logical_total_variation": max(
                        sample["logical_total_variation"] for sample in samples
                    ),
                    "mean_of_unit_top_class_accuracy": float(
                        np.mean([unit["top_class_accuracy"] for unit in unit_summaries])
                    ),
                    "mean_physical_state_coverage": float(
                        np.mean([sample["physical_state_coverage"] for sample in samples])
                    ),
                    "ess_available_count": len(ess_values),
                    "mean_available_logical_ess": (
                        None if not ess_values else float(np.mean(ess_values))
                    ),
                }
            )
    return result


def run_gflownet_study(
    config_path: Path,
    decision_config_path: Path,
    baseline_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Train fixed GFlowNet seeds and evaluate untouched contexts."""
    config = _load_config(config_path)
    decision = json.loads(decision_config_path.read_text())
    _validate_decision_fields(decision)
    baseline = json.loads(baseline_path.read_text())
    canonical_hash = sha256_file(_CANONICAL_CONFIG)
    run_label = (
        "discovery_nonconfirmatory"
        if sha256_file(config_path) == canonical_hash
        else "reduced_non_scientific"
    )
    if baseline["provenance"]["decision_config_sha256"] != sha256_file(decision_config_path):
        raise ValueError("baseline and GFlowNet must use the same decision configuration")
    training_ids = {str(field["id"]) for field in config["training_fields"]}
    training_vectors = {
        tuple(float(value) for value in field["probabilities"])
        for field in config["training_fields"]
    }
    heldout_ids = {str(field["id"]) for field in decision["true_channels"]}
    heldout_vectors = {
        tuple(float(value) for value in field["probabilities"])
        for field in decision["true_channels"]
    }
    if training_ids & heldout_ids or training_vectors & heldout_vectors:
        raise ValueError("training and heldout fields must be identity- and content-disjoint")
    problem = canonical_steane_z_problem()
    training_environments = [
        _environment(field, int(syndrome_index))
        for field in config["training_fields"]
        for syndrome_index in config["training_syndrome_indices"]
    ]
    evaluation_contexts = _evaluation_contexts(config, decision)
    training_records: list[dict[str, object]] = []
    evaluation_rows: list[dict[str, object]] = []

    torch.use_deterministic_algorithms(True)
    for seed in config["training_seeds"]:
        torch.manual_seed(int(seed) % (1 << 63))
        model = TrajectoryBalanceGFlowNet(
            context_width=problem.hx.shape[0] + problem.n,
            coefficient_width=problem.z_stabilizers.shape[0] + 1,
            hidden_width=int(config["hidden_width"]),
        )
        record = train_trajectory_balance(
            model,
            training_environments,
            steps=int(config["training_steps"]),
            learning_rate=float(config["learning_rate"]),
        )
        training_records.append(
            {
                "training_seed": int(seed),
                "initial_loss": record.initial_loss,
                "final_loss": record.final_loss,
                "steps": record.steps,
                "unique_terminal_reward_evaluations": (record.unique_terminal_reward_evaluations),
                "optimization_terminal_terms": record.optimization_terminal_terms,
                "optimization_policy_action_evaluations": (
                    record.optimization_policy_action_evaluations
                ),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            }
        )
        for split, field, syndrome_index in evaluation_contexts:
            evaluation_rows.append(
                _evaluate(
                    model,
                    training_seed=int(seed),
                    split=split,
                    field=field,
                    syndrome_index=syndrome_index,
                    budgets=[int(value) for value in config["sample_budgets"]],
                    sampling_replicate_seeds=[
                        int(value) for value in config["sampling_replicate_seeds"]
                    ],
                )
            )

    payload: dict[str, object] = {
        "schema_version": 1,
        "run_label": run_label,
        "gate4_status": "unresolved_requires_fresh_paired_baselines",
        "claim_boundary": (
            "A discovery-only fixed-order trajectory-balance experiment with exhaustive "
            "development terminal rewards and inspected evaluation contexts on one Steane "
            "code. It cannot pass Gate 4. Fresh fields, repeated paired baselines, and a "
            "frozen development-only selection rule are required before confirmation."
        ),
        "provenance": {
            "config_sha256": sha256_file(config_path),
            "canonical_config_sha256": canonical_hash,
            "decision_config_sha256": sha256_file(decision_config_path),
            "baseline_sha256": sha256_file(baseline_path),
            "source_sha256": {path.name: sha256_file(path) for path in _SOURCE_PATHS},
        },
        "data_boundaries": {
            "training_field_ids": [field["id"] for field in config["training_fields"]],
            "training_syndrome_indices": config["training_syndrome_indices"],
            "heldout_syndrome_indices": config["heldout_syndrome_indices"],
            "heldout_field_ids": [field["id"] for field in decision["true_channels"]],
            "baseline_run_label": baseline["claim_boundary"],
            "sampling_replicate_seeds": config["sampling_replicate_seeds"],
            "wall_clock_status": "not_measured_no_cost_or_latency_comparison_permitted",
        },
        "training_records": training_records,
        "evaluation_rows": evaluation_rows,
        "summaries": _summaries(evaluation_rows),
        "sampled_summaries": _sampled_summaries(evaluation_rows),
        "baseline_summaries": baseline["summaries"],
    }
    write_canonical_json(output_dir / "gflownet_mass.json", payload)
    return payload
