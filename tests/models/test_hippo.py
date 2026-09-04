from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from qldpc_fno.models.hippo import HiPPOLegSMemory, legs_transition

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/models/fixtures/hippo_legs_float64.json"
GENERATOR = ROOT / "tools/generate_hippo_legs_fixture.py"


def _golden() -> dict[str, object]:
    return json.loads(FIXTURE.read_text())


def test_legs_generator_has_official_sign_and_normalization() -> None:
    transition, injection = legs_transition(4, 1, dtype=torch.float64)
    expected_generator = torch.tensor(_golden()["generator"], dtype=torch.float64)
    expected_input = torch.tensor(_golden()["input_vector"], dtype=torch.float64)
    identity = torch.eye(4, dtype=torch.float64)
    lhs = identity - expected_generator / 2

    expected_transition = torch.linalg.solve(lhs, identity + expected_generator / 2)
    expected_injection = torch.linalg.solve(lhs, expected_input)
    torch.testing.assert_close(transition, expected_transition, rtol=0, atol=1e-14)
    torch.testing.assert_close(injection, expected_injection, rtol=0, atol=1e-14)


def test_independent_fixture_regenerates_byte_identically(tmp_path: Path) -> None:
    source = GENERATOR.read_text()
    assert "qldpc_fno" not in source
    generated = tmp_path / "fixture.json"
    subprocess.run(
        [sys.executable, str(GENERATOR), "--output", str(generated)],
        cwd=ROOT,
        check=True,
    )
    assert generated.read_bytes() == FIXTURE.read_bytes()


def test_recurrence_matches_independent_float64_fixture_step_by_step() -> None:
    golden = _golden()
    memory = HiPPOLegSMemory(order=int(golden["order"]), dtype=torch.float64)
    state = torch.zeros(2, 3, int(golden["order"]), dtype=torch.float64)

    for value, record in zip(golden["inputs"], golden["records"], strict=True):
        step = int(record["completed_sample_index"])
        transition, injection = legs_transition(4, step, dtype=torch.float64)
        torch.testing.assert_close(
            transition,
            torch.tensor(record["abar"], dtype=torch.float64),
            rtol=0,
            atol=2e-14,
        )
        torch.testing.assert_close(
            injection,
            torch.tensor(record["bbar"], dtype=torch.float64),
            rtol=0,
            atol=2e-14,
        )
        state = memory.step(torch.full((2, 3), value, dtype=torch.float64), state, step)
        expected = torch.tensor(record["state"], dtype=torch.float64).expand(2, 3, -1)
        torch.testing.assert_close(state, expected, rtol=0, atol=3e-13)


def test_shifted_legendre_reconstruction_converges_on_smooth_history() -> None:
    quadrature_x, quadrature_weight = np.polynomial.legendre.leggauss(512)
    unit_x = (quadrature_x + 1.0) / 2.0
    unit_weight = quadrature_weight / 2.0
    target = np.exp(0.4 * unit_x) + 0.2 * np.sin(2 * np.pi * unit_x)
    order = 16
    basis = np.stack(
        [
            np.sqrt(2 * degree + 1)
            * np.polynomial.legendre.Legendre.basis(degree)(quadrature_x)
            for degree in range(order)
        ]
    )
    direct = (basis * (target * unit_weight)[None, :]).sum(axis=1)
    reconstruction_errors: list[float] = []

    for sample_count in (1024, 4096):
        grid = (torch.arange(sample_count, dtype=torch.float64) + 1) / sample_count
        samples = torch.exp(0.4 * grid) + 0.2 * torch.sin(2 * torch.pi * grid)
        final_state = HiPPOLegSMemory(order=order, dtype=torch.float64)(
            samples[None, :, None], return_sequence=False
        )[0, 0]
        coefficients = final_state.detach().numpy()
        coefficient_tolerance = 3.0 / sample_count
        np.testing.assert_allclose(coefficients, direct, atol=coefficient_tolerance, rtol=0)
        reconstruction = coefficients @ basis
        reconstruction_errors.append(
            float(np.sum(unit_weight * (reconstruction - target) ** 2))
        )

    assert reconstruction_errors[1] < reconstruction_errors[0] * 0.08


@pytest.mark.parametrize("order", [8, 16, 32])
def test_long_recurrence_has_finite_states_and_gradients(order: int) -> None:
    torch.manual_seed(20260904 + order)
    values = (0.1 * torch.randn(1, 4096, 2, 1)).requires_grad_()
    final_state = HiPPOLegSMemory(order=order)(values, return_sequence=False)
    assert final_state.shape == (1, 2, 1, order)
    assert torch.isfinite(final_state).all()

    final_state.square().mean().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


def test_chunked_forward_matches_one_pass_with_completed_sample_offset() -> None:
    torch.manual_seed(20260904)
    values = torch.randn(2, 37, 3, 4, dtype=torch.float64)
    memory = HiPPOLegSMemory(order=8, dtype=torch.float64)

    full = memory(values)
    prefix = memory(values[:, :13])
    suffix = memory(
        values[:, 13:],
        initial_state=prefix[:, -1],
        start_completed_sample_index=13,
    )

    torch.testing.assert_close(torch.cat((prefix, suffix), dim=1), full, rtol=0, atol=1e-13)


def test_forward_rejects_negative_completed_sample_offset() -> None:
    memory = HiPPOLegSMemory(order=4)
    with pytest.raises(ValueError, match="start_completed_sample_index"):
        memory(torch.zeros(1, 2), start_completed_sample_index=-1)


@pytest.mark.parametrize("offset", [1.5, True])
def test_forward_rejects_non_exact_integer_completed_sample_offset(offset: object) -> None:
    memory = HiPPOLegSMemory(order=4)
    with pytest.raises(TypeError, match="start_completed_sample_index"):
        memory(torch.zeros(1, 2), start_completed_sample_index=offset)  # type: ignore[arg-type]


@pytest.mark.parametrize("step", [1.5, True])
def test_transition_rejects_non_exact_integer_step(step: object) -> None:
    with pytest.raises(TypeError, match="step"):
        legs_transition(4, step, dtype=torch.float64)  # type: ignore[arg-type]


@pytest.mark.parametrize("step", [1.5, True])
def test_memory_step_rejects_non_exact_integer_index(step: object) -> None:
    memory = HiPPOLegSMemory(order=4)
    with pytest.raises(TypeError, match="completed_sample_index"):
        memory.step(torch.zeros(2), torch.zeros(2, 4), step)  # type: ignore[arg-type]


def test_transition_rejects_nonpositive_order_or_step() -> None:
    with pytest.raises(ValueError, match="order"):
        legs_transition(0, 1, dtype=torch.float64)
    with pytest.raises(ValueError, match="step"):
        legs_transition(4, 0, dtype=torch.float64)


def test_forward_requires_continuation_state_and_offset_together() -> None:
    memory = HiPPOLegSMemory(order=4)
    values = torch.zeros(2, 3, 5)
    state = torch.zeros(2, 5, 4)

    with pytest.raises(ValueError, match="initial_state.*start_completed_sample_index"):
        memory(values, initial_state=state)
    with pytest.raises(ValueError, match="initial_state.*start_completed_sample_index"):
        memory(values, start_completed_sample_index=7)
