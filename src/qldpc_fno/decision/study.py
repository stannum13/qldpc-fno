"""Deterministic prediction-to-decision sensitivity study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.decision.exact_css import (
    ExactCssPosterior,
    ExactCssProblem,
    bayes_logical_risk,
    canonical_steane_z_problem,
    exact_posterior,
)

_PRIOR_IDS = ("nominal_uniform", "correct_global", "correct_spatial")
_DECODERS = ("physical_map", "coset_map")
_SOURCE_PATHS = (Path(__file__), Path(__file__).with_name("exact_css.py"))


def _load_config(path: Path, n: int) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("prediction-to-decision schema_version must be 1")
    nominal = float(payload["nominal_probability"])
    if not 0.0 < nominal < 0.5:
        raise ValueError("nominal_probability must lie between zero and one half")
    channels = payload.get("true_channels")
    if not isinstance(channels, list) or not channels:
        raise ValueError("true_channels must be a nonempty list")
    seen: set[str] = set()
    for channel in channels:
        if not isinstance(channel, dict) or not isinstance(channel.get("id"), str):
            raise TypeError("every true channel requires a string id")
        if channel["id"] in seen:
            raise ValueError("true channel ids must be unique")
        seen.add(channel["id"])
        probabilities = np.asarray(channel.get("probabilities"), dtype=np.float64)
        if probabilities.shape != (n,) or not np.all((probabilities > 0) & (probabilities < 0.5)):
            raise ValueError(
                "true channel probabilities must contain one value in (0, 0.5) per qubit"
            )
    return payload


def _code_identity(problem: ExactCssProblem) -> dict[str, object]:
    algebra = {
        "hx": problem.hx.tolist(),
        "logical_x": problem.logical_x.tolist(),
        "logical_z": problem.logical_z.tolist(),
        "z_stabilizers": problem.z_stabilizers.tolist(),
    }
    encoded = json.dumps(algebra, sort_keys=True, separators=(",", ":")).encode()
    return {
        "name": problem.name,
        "n": problem.n,
        "checks": int(problem.hx.shape[0]),
        "logical_qubits": 1,
        "identity_sha256": hashlib.sha256(encoded).hexdigest(),
        **algebra,
    }


def _syndrome(index: int, width: int) -> np.ndarray:
    return np.array([(index >> shift) & 1 for shift in range(width)], dtype=np.uint8)


def _assumed_priors(
    true_probabilities: np.ndarray, nominal_probability: float
) -> dict[str, np.ndarray]:
    return {
        "nominal_uniform": np.full(true_probabilities.shape, nominal_probability),
        "correct_global": np.full(true_probabilities.shape, float(true_probabilities.mean())),
        "correct_spatial": np.array(true_probabilities, copy=True),
    }


def _selected_error(posterior: ExactCssPosterior, decoder: str) -> tuple[np.ndarray, int]:
    if decoder == "physical_map":
        return posterior.physical_map_error, posterior.physical_map_class
    selected_class = posterior.coset_map_class
    allowed = np.flatnonzero(posterior.logical_classes == selected_class)
    index = int(allowed[np.argmax(posterior.physical_probabilities[allowed])])
    return posterior.compatible_errors[index], selected_class


def _action_row(
    *,
    channel_id: str,
    syndrome: np.ndarray,
    true_posterior: ExactCssPosterior,
    prior_id: str,
    assumed_probabilities: np.ndarray,
    decoder: str,
    assumed_posterior: ExactCssPosterior,
) -> dict[str, object]:
    selected_error, selected_class = _selected_error(assumed_posterior, decoder)
    syndrome_checks = assumed_posterior.enumerated_error_count
    likelihood_evaluations = len(assumed_posterior.compatible_errors)
    signature_evaluations = len(assumed_posterior.logical_classes)
    class_additions = len(assumed_posterior.logical_classes)
    operation_count = (
        syndrome_checks + likelihood_evaluations + signature_evaluations + class_additions
    )
    return {
        "true_channel": channel_id,
        "syndrome": syndrome.tolist(),
        "syndrome_probability": float(true_posterior.unnormalized_logical_class_masses.sum()),
        "action_id": f"{prior_id}/{decoder}",
        "prior_id": prior_id,
        "decoder": decoder,
        "assumed_probabilities": assumed_probabilities.tolist(),
        "selected_physical_error": selected_error.tolist(),
        "selected_logical_class": selected_class,
        "inferred_logical_class_probabilities": (
            assumed_posterior.logical_class_probabilities.tolist()
        ),
        "true_logical_class_probabilities": true_posterior.logical_class_probabilities.tolist(),
        "true_max_physical_posterior": float(true_posterior.physical_probabilities.max()),
        "conditional_bayes_risk": bayes_logical_risk(selected_class, true_posterior),
        "operation_count": operation_count,
        "operation_breakdown": {
            "syndrome_checks": syndrome_checks,
            "likelihood_evaluations": likelihood_evaluations,
            "logical_signature_evaluations": signature_evaluations,
            "logical_class_additions": class_additions,
        },
    }


def _summaries(
    rows: list[dict[str, object]], config: dict[str, object]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    summaries: list[dict[str, object]] = []
    channel_ids = tuple(dict.fromkeys(str(row["true_channel"]) for row in rows))
    action_ids = tuple(dict.fromkeys(str(row["action_id"]) for row in rows))
    for channel_id in channel_ids:
        channel_rows = [row for row in rows if row["true_channel"] == channel_id]
        expected_by_action: dict[str, float] = {}
        for action_id in action_ids:
            action_rows = [row for row in channel_rows if row["action_id"] == action_id]
            expected_by_action[action_id] = float(
                sum(
                    float(row["syndrome_probability"]) * float(row["conditional_bayes_risk"])
                    for row in action_rows
                )
            )
        baseline = expected_by_action["nominal_uniform/physical_map"]
        best_fixed_id = min(action_ids, key=lambda item: (expected_by_action[item], item))
        prior_effects: list[dict[str, object]] = []
        for prior_id in ("correct_global", "correct_spatial"):
            for decoder in _DECODERS:
                reference = {
                    tuple(row["syndrome"]): row
                    for row in channel_rows
                    if row["action_id"] == f"nominal_uniform/{decoder}"
                }
                candidates = [
                    row for row in channel_rows if row["action_id"] == f"{prior_id}/{decoder}"
                ]
                prior_effects.append(
                    {
                        "prior_id": prior_id,
                        "decoder": decoder,
                        "physical_correction_change_count": sum(
                            row["selected_physical_error"]
                            != reference[tuple(row["syndrome"])]["selected_physical_error"]
                            for row in candidates
                        ),
                        "logical_class_change_count": sum(
                            row["selected_logical_class"]
                            != reference[tuple(row["syndrome"])]["selected_logical_class"]
                            for row in candidates
                        ),
                        "conditional_bayes_risk_change_count": sum(
                            abs(
                                float(row["conditional_bayes_risk"])
                                - float(reference[tuple(row["syndrome"])]["conditional_bayes_risk"])
                            )
                            > float(config["decision_variation_tolerance"])
                            for row in candidates
                        ),
                        "expected_logical_failure_improvement": (
                            expected_by_action[f"nominal_uniform/{decoder}"]
                            - expected_by_action[f"{prior_id}/{decoder}"]
                        ),
                    }
                )
        per_syndrome_oracle = 0.0
        for syndrome_index in range(8):
            syndrome_rows = [
                row
                for row in channel_rows
                if sum(int(bit) << shift for shift, bit in enumerate(row["syndrome"]))
                == syndrome_index
            ]
            per_syndrome_oracle += float(syndrome_rows[0]["syndrome_probability"]) * min(
                float(row["conditional_bayes_risk"]) for row in syndrome_rows
            )
        summaries.append(
            {
                "true_channel": channel_id,
                "expected_logical_failure_by_action": expected_by_action,
                "baseline_action": "nominal_uniform/physical_map",
                "baseline_expected_logical_failure": baseline,
                "best_fixed_action": best_fixed_id,
                "best_fixed_expected_logical_failure": expected_by_action[best_fixed_id],
                "best_fixed_absolute_improvement": baseline - expected_by_action[best_fixed_id],
                "per_syndrome_oracle_expected_logical_failure": per_syndrome_oracle,
                "per_syndrome_oracle_absolute_improvement": baseline - per_syndrome_oracle,
                "prior_effects_against_nominal": prior_effects,
            }
        )

    aggregation_changes = 0
    material_states = 0
    state_keys = tuple(
        dict.fromkeys((str(row["true_channel"]), tuple(row["syndrome"])) for row in rows)
    )
    variation_tolerance = float(config["decision_variation_tolerance"])
    physical_ceiling = float(config["material_configuration_probability_ceiling"])
    logical_floor = float(config["material_logical_class_probability_floor"])
    for channel_id, syndrome_bits in state_keys:
        state_rows = [
            row
            for row in rows
            if row["true_channel"] == channel_id and tuple(row["syndrome"]) == syndrome_bits
        ]
        true_classes = np.asarray(state_rows[0]["true_logical_class_probabilities"])
        if float(state_rows[0]["true_max_physical_posterior"]) < physical_ceiling and np.all(
            true_classes > logical_floor
        ):
            material_states += 1
        for prior_id in _PRIOR_IDS:
            by_decoder = {
                str(row["decoder"]): row for row in state_rows if row["prior_id"] == prior_id
            }
            if (
                by_decoder["physical_map"]["selected_logical_class"]
                != by_decoder["coset_map"]["selected_logical_class"]
            ):
                aggregation_changes += 1

    channel_count = len(channel_ids)
    constant_risks = {
        action_id: sum(
            float(row["syndrome_probability"])
            * float(row["conditional_bayes_risk"])
            / channel_count
            for row in rows
            if row["action_id"] == action_id
        )
        for action_id in action_ids
    }
    best_constant_action = min(action_ids, key=lambda item: (constant_risks[item], item))
    observable_oracle_risk = 0.0
    common_optimal_actions = set(action_ids)
    for syndrome_bits in dict.fromkeys(tuple(row["syndrome"]) for row in rows):
        syndrome_probability = sum(
            float(row["syndrome_probability"]) / channel_count
            for row in rows
            if tuple(row["syndrome"]) == syndrome_bits and row["action_id"] == action_ids[0]
        )
        risk_numerators = {
            action_id: sum(
                float(row["syndrome_probability"])
                * float(row["conditional_bayes_risk"])
                / channel_count
                for row in rows
                if tuple(row["syndrome"]) == syndrome_bits and row["action_id"] == action_id
            )
            for action_id in action_ids
        }
        oracle_action = min(action_ids, key=lambda item: (risk_numerators[item], item))
        observable_oracle_risk += risk_numerators[oracle_action]
        conditional_risks = {
            action_id: numerator / syndrome_probability
            for action_id, numerator in risk_numerators.items()
        }
        minimum_conditional_risk = min(conditional_risks.values())
        common_optimal_actions &= {
            action_id
            for action_id, risk in conditional_risks.items()
            if risk <= minimum_conditional_risk + variation_tolerance
        }
    raw_policy_headroom = constant_risks[best_constant_action] - observable_oracle_risk
    if raw_policy_headroom < -variation_tolerance:
        raise AssertionError("observable oracle cannot be worse than the best constant action")
    policy_headroom = max(0.0, raw_policy_headroom)
    gates = {
        "aggregation_changed_decision_count": aggregation_changes,
        "material_two_class_state_count": material_states,
        "open_probability_mass_inference": aggregation_changes > 0 and material_states > 0,
        "observable_policy_state": "syndrome_only_with_equal_true_channel_mixture",
        "best_constant_action": best_constant_action,
        "best_constant_expected_logical_failure": constant_risks[best_constant_action],
        "observable_policy_common_optimal_actions": sorted(common_optimal_actions),
        "observable_policy_requires_state_dependent_action": not common_optimal_actions,
        "observable_policy_oracle_expected_logical_failure": observable_oracle_risk,
        "observable_policy_oracle_absolute_improvement": policy_headroom,
        "open_learned_action_selection": (
            policy_headroom > variation_tolerance and not common_optimal_actions
        ),
        "adaptive_contraction_requires_separate_local_code_frontier": True,
        "thresholds": {
            "decision_variation_tolerance": variation_tolerance,
            "material_configuration_probability_ceiling": physical_ceiling,
            "material_logical_class_probability_floor": logical_floor,
        },
    }
    return summaries, gates


def run_exact_decision_study(config_path: Path, output_dir: Path) -> dict[str, object]:
    """Run the complete exact study and write one canonical JSON artifact."""
    problem = canonical_steane_z_problem()
    config = _load_config(config_path, problem.n)
    nominal = float(config["nominal_probability"])
    rows: list[dict[str, object]] = []

    for channel in config["true_channels"]:
        channel_id = str(channel["id"])
        true_probabilities = np.asarray(channel["probabilities"], dtype=np.float64)
        priors = _assumed_priors(true_probabilities, nominal)
        for syndrome_index in range(1 << problem.hx.shape[0]):
            measured = _syndrome(syndrome_index, problem.hx.shape[0])
            true_posterior = exact_posterior(problem, measured, true_probabilities)
            for prior_id in _PRIOR_IDS:
                assumed = exact_posterior(problem, measured, priors[prior_id])
                for decoder in _DECODERS:
                    rows.append(
                        _action_row(
                            channel_id=channel_id,
                            syndrome=measured,
                            true_posterior=true_posterior,
                            prior_id=prior_id,
                            assumed_probabilities=priors[prior_id],
                            decoder=decoder,
                            assumed_posterior=assumed,
                        )
                    )

    summaries, gates = _summaries(rows, config)
    payload: dict[str, object] = {
        "schema_version": 1,
        "exact_enumeration": True,
        "claim_boundary": (
            "Exact independent-Z posterior calculation on one Steane-code instance; "
            "not a sampled logical-error rate, threshold, qLDPC, circuit-level, or latency result."
        ),
        "provenance": {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "implementation": "exact_css_enumeration/v1",
            "source_sha256": {path.name: sha256_file(path) for path in _SOURCE_PATHS},
        },
        "code": _code_identity(problem),
        "actions": [f"{prior}/{decoder}" for prior in _PRIOR_IDS for decoder in _DECODERS],
        "operation_count_model": {
            "name": "declared_exact_primitive_count/v1",
            "included": [
                "one parity-check-vector comparison per enumerated physical error",
                "one likelihood evaluation per syndrome-compatible physical error",
                "one logical-signature evaluation per compatible physical error",
                "one class-mass accumulation per compatible physical error",
            ],
            "excluded": ["per-qubit arithmetic", "normalization", "argmax", "allocation"],
            "claim_boundary": (
                "A deterministic algorithmic proxy shared by all exact actions; "
                "not a latency, FLOP, or executed-instruction measurement."
            ),
        },
        "action_table": rows,
        "summaries": summaries,
        "gates": gates,
    }
    write_canonical_json(output_dir / "action_table.json", payload)
    return payload
