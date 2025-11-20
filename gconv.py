import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicRoutingLayer(nn.Module):
    """Generates routing coefficients for dynamic convolution kernels."""

    def __init__(self, in_channels: int, num_kernels: int, device: str = "cuda") -> None:
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_kernels)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.global_avg_pool(x).view(x.size(0), -1)
        logits = self.fc(pooled)
        return F.softmax(logits, dim=1)


class GConv2d(nn.Module):
    """
    P4-equivariant convolution with dynamic kernels.

    The layer learns a bank of kernels and uses input-dependent routing
    coefficients to form sample-specific filters. Each kernel is rotated
    to produce four orientation channels (0, 90, 180, 270 degrees).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        in_transformations: int = 1,
        out_transformations: int = 4,
        device: str = "cuda",
        num_dynamic_kernels: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels * in_transformations
        self.out_channels = out_channels
        self.out_transformations = out_transformations
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.device = device

        self.routing = DynamicRoutingLayer(self.in_channels, num_dynamic_kernels, device=device)
        self.kernel_bank = nn.Parameter(
            torch.randn(num_dynamic_kernels, out_channels, self.in_channels, filter_size, filter_size, device=device)
        )
        self.to(device)

    def _expand_orientations(self, kernels: torch.Tensor) -> torch.Tensor:
        rotated = [torch.rot90(kernels, k, dims=(-2, -1)) for k in range(self.out_transformations)]
        return torch.cat(rotated, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routing_coeffs = self.routing(x)
        base_kernels = torch.einsum("bk,kocij->bocij", routing_coeffs, self.kernel_bank)
        oriented_kernels = self._expand_orientations(base_kernels)

        outputs = []
        for sample, kernels in zip(x, oriented_kernels):
            outputs.append(
                F.conv2d(
                    sample.unsqueeze(0),
                    kernels,
                    stride=self.stride,
                    padding=self.padding,
                )
            )
        return torch.cat(outputs, dim=0).to(self.device)


__all__ = ["GConv2d"]
