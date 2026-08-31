import pytest
import torch

from qldpc_fno.models.fno1d import RingFNO


def test_ring_fno_preserves_ring_length_and_outputs_corrections() -> None:
    model = RingFNO(in_channels=21, out_channels=58, width=16, modes=8, depth=2)
    inputs = torch.randn(4, 21, 45)
    outputs = model(inputs)
    assert outputs.shape == (4, 58, 45)
    outputs.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_modes_cannot_exceed_rfft_width() -> None:
    model = RingFNO(in_channels=2, out_channels=3, width=4, modes=20, depth=1)
    with pytest.raises(ValueError, match="modes"):
        model(torch.zeros(1, 2, 9))
