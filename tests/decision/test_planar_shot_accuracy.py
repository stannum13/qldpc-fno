from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import qecsim
import scipy
from qecsim import paulitools as pt
from qecsim.models.planar import PlanarCode, PlanarMPSDecoder

from qldpc_fno.decision import planar_shot_accuracy as accuracy
from qldpc_fno.decision.planar_shot_accuracy import (
    calibrate_cmwpm,
    cmwpm_grid,
    logical_class_recovery,
    score_recovery,
    select_cmwpm_candidate,
    two_view_tolerance_decision,
)
from qldpc_fno.decision.planar_shot_data import generate_planar_shots
from qldpc_fno.decision.tensor_network import InvalidCosetMassError, PlanarCosetMasses


def test_class_recoveries_reproduce_syndrome_and_are_stabilizer_invariant() -> None:
    code = PlanarCode(3, 3)
    error = np.zeros(2 * code.n_k_d[0], dtype=np.uint8)
    error[0] = 1
    syndrome = pt.bsp(error, code.stabilizers.T)

    for logical_class in range(4):
        recovery = logical_class_recovery(code, syndrome, logical_class)
        np.testing.assert_array_equal(pt.bsp(recovery, code.stabilizers.T), syndrome)

        scored = score_recovery(code, error, syndrome, recovery)
        assert scored["syndrome_valid"] is True
        assert scored["logical_failure"] == bool(
            np.any(pt.bsp((error + recovery) % 2, code.logicals.T))
        )
        np.testing.assert_array_equal(scored["residual_syndrome"], np.zeros_like(syndrome))

        for stabilizer in code.stabilizers:
            stabilized = (recovery + stabilizer) % 2
            stabilized_scored = score_recovery(code, error, syndrome, stabilized)
            assert stabilized_scored["syndrome_valid"] is True
            assert stabilized_scored["logical_failure"] == scored["logical_failure"]
            np.testing.assert_array_equal(
                stabilized_scored["logical_signature"], scored["logical_signature"]
            )


def test_logical_class_recoveries_match_qecsim_representative_order() -> None:
    code = PlanarCode(3, 3)
    error = np.zeros(2 * code.n_k_d[0], dtype=np.uint8)
    error[0] = 1
    syndrome = pt.bsp(error, code.stabilizers.T)
    sample = PlanarMPSDecoder.sample_recovery(code, syndrome)
    expected = (
        sample,
        sample.copy().logical_x(),
        sample.copy().logical_x().logical_z(),
        sample.copy().logical_z(),
    )

    for logical_class, representative in enumerate(expected):
        recovery = logical_class_recovery(code, syndrome, logical_class)
        np.testing.assert_array_equal(recovery, representative.to_bsf())


@pytest.mark.parametrize(
    "syndrome",
    (np.zeros(11, dtype=np.uint8), np.full(12, 2, dtype=np.uint8)),
)
def test_logical_class_recovery_rejects_invalid_syndrome(syndrome: np.ndarray) -> None:
    with pytest.raises(ValueError, match="syndrome"):
        logical_class_recovery(PlanarCode(3, 3), syndrome, 0)


@pytest.mark.parametrize("logical_class", (-1, 4, True, 1.0))
def test_logical_class_recovery_rejects_invalid_class(logical_class: object) -> None:
    code = PlanarCode(3, 3)
    syndrome = np.zeros(code.stabilizers.shape[0], dtype=np.uint8)

    with pytest.raises(ValueError, match="logical_class"):
        logical_class_recovery(code, syndrome, logical_class)  # type: ignore[arg-type]


def test_logical_shift_is_syndrome_valid_but_changes_failure_outcome() -> None:
    code = PlanarCode(3, 3)
    error = np.zeros(2 * code.n_k_d[0], dtype=np.uint8)
    syndrome = pt.bsp(error, code.stabilizers.T)

    reference = score_recovery(code, error, syndrome, logical_class_recovery(code, syndrome, 0))
    logical_shift = score_recovery(code, error, syndrome, logical_class_recovery(code, syndrome, 1))

    assert reference["syndrome_valid"] is True
    assert logical_shift["syndrome_valid"] is True
    assert reference["logical_failure"] is False
    assert logical_shift["logical_failure"] is True


def test_score_recovery_reports_a_syndrome_invalid_recovery() -> None:
    code = PlanarCode(3, 3)
    error = np.zeros(2 * code.n_k_d[0], dtype=np.uint8)
    error[0] = 1
    syndrome = pt.bsp(error, code.stabilizers.T)

    scored = score_recovery(code, error, syndrome, np.zeros_like(error))

    assert scored["syndrome_valid"] is False
    np.testing.assert_array_equal(scored["recovery_syndrome"], np.zeros_like(syndrome))
    np.testing.assert_array_equal(scored["residual_syndrome"], syndrome)


@pytest.mark.parametrize(
    ("argument", "value"),
    (
        ("error", np.zeros(17, dtype=np.uint8)),
        ("syndrome", np.zeros(11, dtype=np.uint8)),
        ("recovery", np.full(18, 2, dtype=np.uint8)),
    ),
)
def test_score_recovery_rejects_invalid_shapes_and_nonbinary_values(
    argument: str, value: np.ndarray
) -> None:
    code = PlanarCode(3, 3)
    error = np.zeros(2 * code.n_k_d[0], dtype=np.uint8)
    syndrome = pt.bsp(error, code.stabilizers.T)
    recovery = logical_class_recovery(code, syndrome, 0)
    arguments: dict[str, np.ndarray] = {
        "error": error,
        "syndrome": syndrome,
        "recovery": recovery,
    }
    arguments[argument] = value

    with pytest.raises(ValueError):
        score_recovery(code, **arguments)


def _coset_masses(
    *,
    selected_class: int,
    mode: str,
    chi: int | None,
    tol: float | None,
    flops: int,
) -> PlanarCosetMasses:
    return PlanarCosetMasses(
        masses=np.array([0.4, 0.3, 0.2, 0.1]),
        selected_class=selected_class,
        rows=5,
        columns=5,
        error_rate=0.1,
        chi=chi,
        tol=tol,
        mode=mode,
        work={"estimated_arithmetic_flops": flops},
    )


def test_two_view_tolerance_accepts_agreement_and_accounts_for_both_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_masses(**kwargs: object) -> PlanarCosetMasses:
        calls.append(kwargs)
        return _coset_masses(
            selected_class=2,
            mode=str(kwargs["mode"]),
            chi=kwargs["chi"] if isinstance(kwargs["chi"], int) else None,
            tol=kwargs["tol"] if isinstance(kwargs["tol"], float) else None,
            flops=10 if kwargs["mode"] == "columns" else 20,
        )

    monkeypatch.setattr(
        "qldpc_fno.decision.planar_shot_accuracy.planar_mps_coset_masses", fake_masses
    )

    decision = two_view_tolerance_decision(np.zeros(40, dtype=np.uint8), 0.1)

    assert [call["mode"] for call in calls] == ["columns", "rows"]
    assert all(call["tol"] == 0.01 for call in calls)
    assert all(call["chi"] is None for call in calls)
    assert decision["selected_class"] == 2
    assert decision["accepted"] is True
    assert decision["used_fallback"] is False
    assert decision["estimated_arithmetic_flops"] == 30


def test_two_view_tolerance_falls_back_to_column_chi_eight_and_accounts_for_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_masses(**kwargs: object) -> PlanarCosetMasses:
        calls.append(kwargs)
        selected_class = 0 if kwargs["mode"] == "columns" and kwargs["chi"] is None else 1
        return _coset_masses(
            selected_class=selected_class,
            mode=str(kwargs["mode"]),
            chi=kwargs["chi"] if isinstance(kwargs["chi"], int) else None,
            tol=kwargs["tol"] if isinstance(kwargs["tol"], float) else None,
            flops=(10, 20, 30)[len(calls) - 1],
        )

    monkeypatch.setattr(
        "qldpc_fno.decision.planar_shot_accuracy.planar_mps_coset_masses", fake_masses
    )

    decision = two_view_tolerance_decision(np.zeros(40, dtype=np.uint8), 0.1)

    assert [(call["mode"], call["chi"], call["tol"]) for call in calls] == [
        ("columns", None, 0.01),
        ("rows", None, 0.01),
        ("columns", 8, None),
    ]
    assert decision["selected_class"] == 1
    assert decision["accepted"] is False
    assert decision["used_fallback"] is True
    assert decision["estimated_arithmetic_flops"] == 60


@pytest.mark.parametrize(
    ("syndrome", "error_rate"),
    (
        (np.zeros(39, dtype=np.uint8), 0.1),
        (np.full(40, 2, dtype=np.uint8), 0.1),
        (np.full(40, 0.5, dtype=np.float64), 0.1),
        (np.zeros(40, dtype=np.uint8), 0.0),
        (np.zeros(40, dtype=np.uint8), 1.0),
        (np.zeros(40, dtype=np.uint8), float("nan")),
    ),
)
def test_two_view_tolerance_rejects_invalid_policy_inputs(
    syndrome: np.ndarray, error_rate: float
) -> None:
    with pytest.raises(ValueError):
        two_view_tolerance_decision(syndrome, error_rate)


def _cmwpm_grid(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "factors": [1, 2, 3, 4],
                "max_iterations": [2, 4, 8],
                "box_shapes": ["t", "r"],
                "distance_algorithms": [2, 4],
            }
        )
    )
    return path


def _shot_config(tmp_path: Path, domain: str) -> Path:
    config = {
        "schema_version": 1,
        "seed_domain": domain,
        "campaign_seed": int.from_bytes(hashlib.sha256(domain.encode()).digest()[:8], "big"),
        "code_distance": 5,
        "error_rates": [0.1, 0.15],
        "shots_per_rate": 1,
        "noise_model": "qecsim_iid_depolarizing_code_capacity",
    }
    path = tmp_path / "shots.json"
    path.write_text(json.dumps(config))
    return path


def _candidate(
    parameters: tuple[int, int, str, int], *, failures: tuple[int, int]
) -> dict[str, object]:
    factor, max_iterations, box_shape, distance_algorithm = parameters
    return {
        "parameters": {
            "factor": factor,
            "max_iterations": max_iterations,
            "box_shape": box_shape,
            "distance_algorithm": distance_algorithm,
        },
        "per_rate": [
            {"error_rate": 0.1, "failures": failures[0]},
            {"error_rate": 0.15, "failures": failures[1]},
        ],
        "pooled": {"failures": sum(failures)},
    }


def test_cmwpm_selection_uses_pooled_then_worst_rate_then_parameter_tuple() -> None:
    pooled_loser = _candidate((1, 2, "t", 2), failures=(0, 3))
    worst_rate_loser = _candidate((1, 2, "t", 4), failures=(0, 2))
    lexical_loser = _candidate((2, 2, "r", 2), failures=(1, 1))
    expected = _candidate((1, 4, "r", 4), failures=(1, 1))

    assert (
        select_cmwpm_candidate([pooled_loser, worst_rate_loser, lexical_loser, expected])
        == expected
    )


def test_cmwpm_grid_is_frozen_to_one_shared_48_candidate_policy(tmp_path: Path) -> None:
    grid = cmwpm_grid(_cmwpm_grid(tmp_path / "grid.json"))

    assert len(grid) == 48
    assert len({tuple(candidate.values()) for candidate in grid}) == 48
    assert grid[0] == {
        "factor": 1,
        "max_iterations": 2,
        "box_shape": "t",
        "distance_algorithm": 2,
    }


def test_cmwpm_calibration_records_one_shared_selection_and_mwpm_reference(
    tmp_path: Path,
) -> None:
    shots_dir = tmp_path / "calibration-shots"
    shot_config = _shot_config(tmp_path, "qldpc-fno/planar-shot-test/calibration/v1")
    generate_planar_shots(shot_config, shots_dir)
    grid_path = _cmwpm_grid(tmp_path / "grid.json")

    result = calibrate_cmwpm(
        grid_path,
        shots_dir / "planar_shots.json",
        tmp_path / "selection",
    )

    assert len(result["candidates"]) == 48
    assert result["selected"] in result["candidates"]
    assert len(result["selected"]["per_rate"]) == 2
    assert result["mwpm_reference"] not in result["candidates"]
    selection_path = tmp_path / "selection" / "planar_cmwpm_selection.json"
    selection = json.loads(selection_path.read_text())
    assert selection["grid"] == json.loads(grid_path.read_text())
    assert selection["calibration_data_config"] == json.loads(shot_config.read_text())
    provenance = selection["provenance"]
    assert provenance["dependencies"] == {
        "numpy": np.__version__,
        "python": provenance["dependencies"]["python"],
        "qecsim": qecsim.__version__,
        "scipy": scipy.__version__,
    }
    assert isinstance(provenance["git_commit"], str)
    assert isinstance(provenance["git_dirty"], bool)
    source_hashes = provenance["source_sha256"]
    root = Path(__file__).resolve().parents[2]
    assert set(source_hashes) == {
        "experiments/32_calibrate_planar_cmwpm.py",
        "src/qldpc_fno/decision/planar_shot_accuracy.py",
        "src/qldpc_fno/decision/planar_shot_data.py",
    }
    assert source_hashes == {
        label: hashlib.sha256((root / label).read_bytes()).hexdigest() for label in source_hashes
    }


def test_cmwpm_calibration_rejects_missing_data_provenance(tmp_path: Path) -> None:
    shots_dir = tmp_path / "shots"
    generate_planar_shots(
        _shot_config(tmp_path, "qldpc-fno/planar-shot-test/calibration/v1"), shots_dir
    )
    artifact_path = shots_dir / "planar_shots.json"
    artifact = json.loads(artifact_path.read_text())
    del artifact["provenance"]
    artifact_path.write_text(json.dumps(artifact))

    with pytest.raises(TypeError, match="provenance"):
        calibrate_cmwpm(_cmwpm_grid(tmp_path / "grid.json"), artifact_path, tmp_path / "out")


def test_cmwpm_calibration_rejects_malformed_data_provenance(tmp_path: Path) -> None:
    shots_dir = tmp_path / "shots"
    generate_planar_shots(
        _shot_config(tmp_path, "qldpc-fno/planar-shot-test/calibration/v1"), shots_dir
    )
    artifact_path = shots_dir / "planar_shots.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["provenance"]["source_sha256"]["src/qldpc_fno/decision/planar_shot_data.py"] = (
        "not-a-sha256"
    )
    artifact_path.write_text(json.dumps(artifact))

    with pytest.raises(ValueError, match="provenance"):
        calibrate_cmwpm(_cmwpm_grid(tmp_path / "grid.json"), artifact_path, tmp_path / "out")


@pytest.mark.parametrize("domain", ["qldpc-fno/planar-shot-test/screen/v1"])
def test_cmwpm_calibration_rejects_screen_domain(tmp_path: Path, domain: str) -> None:
    shots_dir = tmp_path / "shots"
    generate_planar_shots(_shot_config(tmp_path, domain), shots_dir)

    with pytest.raises(ValueError, match="calibration"):
        calibrate_cmwpm(
            _cmwpm_grid(tmp_path / "grid.json"), shots_dir / "planar_shots.json", tmp_path / "out"
        )


def test_cmwpm_calibration_rejects_unreplayable_physical_error(tmp_path: Path) -> None:
    shots_dir = tmp_path / "shots"
    generate_planar_shots(
        _shot_config(tmp_path, "qldpc-fno/planar-shot-test/calibration/v1"), shots_dir
    )
    artifact_path = shots_dir / "planar_shots.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["shots"][0]["error_bsf"][0] ^= 1
    artifact_path.write_text(json.dumps(artifact))

    with pytest.raises(ValueError, match="replay"):
        calibrate_cmwpm(_cmwpm_grid(tmp_path / "grid.json"), artifact_path, tmp_path / "out")


def test_cmwpm_calibration_counts_decoder_exceptions_and_invalid_recoveries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class BrokenCMWPMDecoder:
        def __init__(self, **_: object) -> None:
            self.calls = 0

        def decode(self, _: PlanarCode, __: np.ndarray) -> np.ndarray:
            self.calls += 1
            if self.calls == 1:
                return np.zeros(1, dtype=np.uint8)
            raise RuntimeError("decoder failed")

    shots_dir = tmp_path / "shots"
    generate_planar_shots(
        _shot_config(tmp_path, "qldpc-fno/planar-shot-test/calibration/v1"), shots_dir
    )
    monkeypatch.setattr(
        "qldpc_fno.decision.planar_shot_accuracy.PlanarCMWPMDecoder", BrokenCMWPMDecoder
    )

    result = calibrate_cmwpm(
        _cmwpm_grid(tmp_path / "grid.json"),
        shots_dir / "planar_shots.json",
        tmp_path / "selection",
    )

    for candidate in result["candidates"]:
        pooled = candidate["pooled"]
        assert pooled["failures"] == 2
        assert pooled["invalid_recoveries"] == 1
        assert pooled["decode_exceptions"] == 1


def test_gate_adjusted_wilson_and_zero_mismatch_requirement() -> None:
    assert accuracy.adjusted_wilson_upper(0, 2048, alpha=0.05 / 2) < 0.005
    assert accuracy.status_for_strata([0, 0], shots=2048) == "passed_exact_outcome_preservation"
    assert accuracy.status_for_strata([0, 1], shots=2048) == "falsified_exact_outcome_preservation"
    assert accuracy.status_for_strata([0, 0], shots=2) == "unresolved_insufficient_precision"


@pytest.mark.parametrize(
    "probabilities",
    [
        [0.4, 0.4, 0.1, 0.1],
        [0.7, 0.2, 0.1, 0.0],
        [float("nan"), 0.2, 0.2, 0.2],
    ],
)
def test_screen_reference_rejects_tied_or_invalid_masses(probabilities: list[float]) -> None:
    with pytest.raises(ValueError, match="reference"):
        accuracy.certify_reference(np.array(probabilities), np.array(probabilities))


def test_screen_reference_requires_both_numerical_metrics_and_robust_margin() -> None:
    good = np.array([0.4, 0.3, 0.2, 0.1])
    assert accuracy.certify_reference(good, good)["selected_class"] == 0
    with pytest.raises(ValueError, match="reference"):
        accuracy.certify_reference(good, good + np.array([1e-9, -1e-9, 0, 0]))
    tiny = np.array([0.6, 0.3, 0.1 - 1e-12, 1e-12])
    with pytest.raises(ValueError, match="reference"):
        accuracy.certify_reference(tiny, tiny + np.array([0, 0, -1e-13, 1e-13]))
    near_tie = np.array([0.4 + 1e-12, 0.4, 0.1, 0.1])
    with pytest.raises(ValueError, match="reference"):
        accuracy.certify_reference(near_tie, near_tie + np.array([0, 0, 1e-12, -1e-12]))


@pytest.fixture
def reduced_screen_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    calibration_config = _shot_config(tmp_path, "qldpc-fno/planar-shot-test/calibration/v1")
    value = json.loads(calibration_config.read_text())
    value["shots_per_rate"] = 2
    calibration_config.write_text(json.dumps(value))
    generate_planar_shots(calibration_config, tmp_path / "calibration")
    grid_path = _cmwpm_grid(tmp_path / "grid.json")
    candidates = cmwpm_grid(grid_path)[:2]
    with monkeypatch.context() as patch:
        patch.setattr(accuracy, "cmwpm_grid", lambda _: candidates)
        calibrate_cmwpm(
            grid_path, tmp_path / "calibration/planar_shots.json", tmp_path / "selection"
        )
    screen_config = _shot_config(tmp_path, "qldpc-fno/planar-shot-test/screen/v1")
    value = json.loads(screen_config.read_text())
    value["shots_per_rate"] = 2
    screen_config.write_text(json.dumps(value))
    generate_planar_shots(screen_config, tmp_path / "screen")
    return (
        Path(__file__).resolve().parents[2] / "configs/planar_shot_accuracy_policy.json",
        tmp_path / "screen/planar_shots.json",
        tmp_path / "selection/planar_cmwpm_selection.json",
    )


def test_screen_reduced_pipeline_paired_tables_work_and_relative_provenance(
    reduced_screen_inputs: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    result = accuracy.run_planar_shot_accuracy(*reduced_screen_inputs, tmp_path / "result")
    assert result["status"] == "unresolved_noncanonical_data"
    assert result["canonical"] is False
    assert result["integrity"]["independent_replay"] is True
    assert len(result["shots"]) == 4
    for row in result["shots"]:
        assert set(row["arms"]) == {"exact", "policy", "cmwpm", "mwpm"}
        assert all(arm["syndrome_valid"] for arm in row["arms"].values())
        assert row["work"]["policy"] > 0
        assert row["work"]["exact"] > 0
    for stratum in result["per_rate"]:
        paired = stratum["paired"]["exact_vs_cmwpm"]
        rows = [row for row in result["shots"] if row["error_rate"] == stratum["error_rate"]]
        exact = np.array([row["arms"]["exact"]["failure"] for row in rows])
        matching = np.array([row["arms"]["cmwpm"]["failure"] for row in rows])
        assert paired["baseline_only_failure"] == int(np.sum(exact & ~matching))
        assert paired["hybrid_only_failure"] == int(np.sum(~exact & matching))
        assert paired["both_fail"] == int(np.sum(exact & matching))
    assert result["work"]["bootstrap_replicates"] == 10000
    assert result["work"]["bootstrap_seed"] == 13158893872079179326
    assert all(not Path(label).is_absolute() for label in result["provenance"]["source_sha256"])
    assert str(tmp_path) not in json.dumps(result)
    assert json.loads((tmp_path / "result/planar_shot_accuracy.json").read_text()) == result
    with pytest.raises(FileExistsError):
        accuracy.run_planar_shot_accuracy(*reduced_screen_inputs, tmp_path / "result")


@pytest.mark.parametrize("tamper", ["error", "selection", "collision", "stale_source"])
def test_screen_integrity_rejects_tampering(
    reduced_screen_inputs: tuple[Path, Path, Path],
    tmp_path: Path,
    tamper: str,
) -> None:
    policy, screen, selection = reduced_screen_inputs
    path = screen if tamper in {"error", "collision"} else selection
    payload = json.loads(path.read_text())
    if tamper == "error":
        payload["shots"][0]["error_bsf"][0] ^= 1
    elif tamper == "selection":
        payload["selected"]["pooled"]["failures"] += 1
    elif tamper == "collision":
        payload["config"] = json.loads(selection.read_text())["calibration_data_config"]
    else:
        label = next(iter(payload["provenance"]["source_sha256"]))
        payload["provenance"]["source_sha256"][label] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="replay|selection|domain|source"):
        accuracy.run_planar_shot_accuracy(policy, screen, selection, tmp_path / "result")


def test_screen_integrity_detects_stored_result_tampering(
    reduced_screen_inputs: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    result = accuracy.run_planar_shot_accuracy(*reduced_screen_inputs, tmp_path / "result")
    result["shots"][0]["work"]["policy"] += 1
    with pytest.raises(ValueError, match="replay"):
        accuracy.verify_screen_result(result, *reduced_screen_inputs)


def test_screen_policy_invalid_view_still_runs_both_views_and_charges_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def tensor(**kwargs: object) -> PlanarCosetMasses:
        calls.append((kwargs["mode"], kwargs["chi"]))
        if len(calls) == 1:
            raise InvalidCosetMassError(
                np.array([float("nan")] * 4), {"estimated_arithmetic_flops": 11}
            )
        return _coset_masses(
            selected_class=0,
            mode=str(kwargs["mode"]),
            chi=kwargs["chi"],
            tol=kwargs["tol"],
            flops=17,
        )

    monkeypatch.setattr(accuracy, "planar_mps_coset_masses", tensor)
    result = accuracy.two_view_tolerance_decision(np.zeros(40, dtype=np.uint8), 0.1)
    assert calls == [("columns", None), ("rows", None), ("columns", 8)]
    assert result["used_fallback"] is True
    assert result["estimated_arithmetic_flops"] == 45
    assert result["invalid_views"] == ["columns"]


def test_screen_gate_rejects_class_mismatch_even_when_both_logically_fail(
    reduced_screen_inputs: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    result = accuracy.run_planar_shot_accuracy(*reduced_screen_inputs, tmp_path / "result")
    row = result["shots"][0]
    row["class_mismatch"] = True
    row["outcome_discordance"] = False
    assert (
        accuracy._summarize_screen(result["shots"], canonical=True)["status"]
        == "falsified_exact_outcome_preservation"
    )
    row["class_mismatch"] = False
    row["outcome_discordance"] = True
    assert (
        accuracy._summarize_screen(result["shots"], canonical=True)["status"]
        == "falsified_exact_outcome_preservation"
    )
    row["reference_valid"] = False
    assert (
        accuracy._summarize_screen(result["shots"], canonical=True)["status"]
        == "invalid_exact_reference"
    )
    assert (
        accuracy._summarize_screen(result["shots"], canonical=False)["status"]
        == "unresolved_noncanonical_data"
    )


@pytest.mark.parametrize(
    "field",
    ["class", "probability", "signature", "aggregate", "data_hash", "selection_hash", "parameters"],
)
def test_screen_integrity_recomputes_stored_scientific_fields(
    reduced_screen_inputs: tuple[Path, Path, Path],
    tmp_path: Path,
    field: str,
) -> None:
    result = accuracy.run_planar_shot_accuracy(*reduced_screen_inputs, tmp_path / "result")
    row = result["shots"][0]
    if field == "class":
        row["exact_class"] = (row["exact_class"] + 1) % 4
    elif field == "probability":
        row["tensor"]["exact_columns"]["probabilities"][0] += 0.1
    elif field == "signature":
        row["arms"]["policy"]["logical_signature"][0] ^= 1
    elif field == "aggregate":
        result["per_rate"][0]["class_mismatches"] += 1
    elif field in {"data_hash", "selection_hash"}:
        result["provenance"]["input_sha256"]["screen" if field == "data_hash" else "selection"] = (
            "0" * 64
        )
    else:
        result["selected_cmwpm_parameters"]["factor"] += 1
    with pytest.raises(ValueError, match="replay"):
        accuracy.verify_screen_result(result, *reduced_screen_inputs)


def test_screen_invalid_matching_recovery_is_failure_and_integrity_error(
    reduced_screen_inputs: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidDecoder:
        def decode(self, code: PlanarCode, syndrome: np.ndarray) -> np.ndarray:
            recovery = logical_class_recovery(code, syndrome, 0)
            recovery[0] ^= 1
            return recovery

    monkeypatch.setattr(accuracy, "PlanarMWPMDecoder", InvalidDecoder)
    result = accuracy.run_planar_shot_accuracy(*reduced_screen_inputs, tmp_path / "result")
    assert all(row["arms"]["mwpm"]["failure"] for row in result["shots"])
    assert all(not row["arms"]["mwpm"]["syndrome_valid"] for row in result["shots"])
    assert (
        accuracy._summarize_screen(result["shots"], canonical=True)["status"] == "invalid_recovery"
    )


@pytest.mark.parametrize("action", ["exact", "fallback"])
def test_screen_invalid_tensor_actions_preserve_diagnostics_and_work(
    reduced_screen_inputs: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    real_tensor = accuracy.planar_mps_coset_masses

    def tensor(**kwargs: object) -> PlanarCosetMasses:
        if (action == "exact" and kwargs["chi"] is None and kwargs["tol"] is None) or (
            action == "fallback" and (kwargs["chi"] == 8 or kwargs["tol"] == 0.01)
        ):
            raise InvalidCosetMassError(
                np.array([float("nan")] * 4), {"estimated_arithmetic_flops": 13}
            )
        return real_tensor(**kwargs)

    monkeypatch.setattr(accuracy, "planar_mps_coset_masses", tensor)
    result = accuracy.run_planar_shot_accuracy(*reduced_screen_inputs, tmp_path / "result")
    for row in result["shots"]:
        if action == "exact":
            assert row["reference_valid"] is False
            assert row["work"]["exact"] == 26
        else:
            assert row["arms"]["policy"]["failure"] is True
            assert row["work"]["policy"] == 39
            assert row["work"]["fixed_chi8"] == 13
    assert result["status"] == "unresolved_noncanonical_data"
    assert result["integrity"]["independent_replay"] is True


def test_screen_work_bootstrap_uses_paired_equal_rate_means() -> None:
    rows = [
        {
            "error_rate": rate,
            "work": {"policy": factor * 2, "fixed_chi8": factor * 4, "exact": factor * 8},
        }
        for rate in (0.1, 0.15)
        for factor in (1, 3)
    ]
    result = accuracy._work_summary(rows)
    assert result["policy_relative_to"]["fixed_chi8"]["ratio"] == 0.5
    assert result["policy_relative_to"]["fixed_chi8"]["ratio_95ci"] == [0.5, 0.5]
    assert result["policy_relative_to"]["exact"]["ratio_95ci"] == [0.25, 0.25]
    assert accuracy._work_summary(rows) == result
