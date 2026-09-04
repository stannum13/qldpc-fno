from __future__ import annotations

import torch
from torch import nn


class SpectralConv1d(nn.Module):
    """Learned channel mixing on a fixed prefix of cyclic Fourier modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or modes <= 0:
            raise ValueError("in_channels, out_channels, and modes must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(values, dim=-1)
        if self.modes > spectrum.shape[-1]:
            raise ValueError(f"modes={self.modes} exceeds rFFT width={spectrum.shape[-1]}")
        output = torch.zeros(
            values.shape[0],
            self.out_channels,
            spectrum.shape[-1],
            dtype=spectrum.dtype,
            device=values.device,
        )
        output[..., : self.modes] = torch.einsum(
            "bim,iom->bom", spectrum[..., : self.modes], self.weight
        )
        return torch.fft.irfft(output, n=values.shape[-1], dim=-1)


class FNOBlock1d(nn.Module):
    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.local = nn.Conv1d(width, width, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(self.spectral(values) + self.local(values))


class RingFNO(nn.Module):
    """Small translation-equivariant neural operator over one cyclic coordinate."""

    def __init__(
        self,
        in_channels: int = 21,
        out_channels: int = 58,
        width: int = 32,
        modes: int = 12,
        depth: int = 2,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes = modes
        self.depth = depth
        self.lift = nn.Conv1d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList(FNOBlock1d(width, modes) for _ in range(depth))
        self.project = nn.Conv1d(width, out_channels, kernel_size=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("values must have shape (batch, channels, ring length)")
        if values.shape[1] != self.in_channels:
            raise ValueError(
                f"input has {values.shape[1]} channels; expected {self.in_channels}"
            )
        hidden = self.lift(values)
        for block in self.blocks:
            hidden = block(hidden)
        return self.project(hidden)

    def configuration(self) -> dict[str, int]:
        return {
            "depth": self.depth,
            "in_channels": self.in_channels,
            "modes": self.modes,
            "out_channels": self.out_channels,
            "width": self.width,
        }


class RingFNOEncoder(nn.Module):
    """Translation-equivariant FNO feature encoder without an output projection."""

    def __init__(
        self,
        in_channels: int = 21,
        width: int = 32,
        modes: int = 12,
        depth: int = 2,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.in_channels = in_channels
        self.width = width
        self.modes = modes
        self.depth = depth
        self.lift = nn.Conv1d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList(FNOBlock1d(width, modes) for _ in range(depth))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("values must have shape (batch, channels, ring length)")
        if values.shape[1] != self.in_channels:
            raise ValueError(
                f"input has {values.shape[1]} channels; expected {self.in_channels}"
            )
        hidden = self.lift(values)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden
