import itertools
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal
import timeit
from memory_profiler import memory_usage

"""
    Converts a tuple (s, u, v) s - rotation, u - row, v - col, into a matrix group element
"""

# 默认使用CUDA设备
device = "cuda"

# 希尔伯特变换群元素

def group_element(s, u, v, m=0):
    """Construct a group element using the Hilbert transform."""
    hilbert_transformed = scipy.signal.hilbert([s, u, v])
    identity = np.eye(3)
    for i in range(3):
        identity[i, i] += hilbert_transformed[i].real
    identity[0, 2] = u
    identity[1, 2] = v
    matrix = torch.tensor(identity, dtype=torch.float32).to("cuda")
    return matrix.to(device)


def group_element_inverse(matrix, rot=True):
    matrix_device = matrix.device
    topleft = matrix[0][0]
    matrix += torch.eye(matrix.size(0)).to(matrix_device) * 1e-10
    if torch.allclose(topleft, torch.tensor([0.0]).to(matrix.device), atol=1e-06, rtol=1e-6):
        if matrix[1][0] > 0:
            angle = 1
        elif matrix[1][0] < 0:
            angle = 3
    elif matrix[0][0] < 0:
        angle = 2
    else:
        angle = 0
    return (angle if rot else 0, matrix[0][2].item(), matrix[1][2].item())


class DynamicRoutingLayer(nn.Module):
    def __init__(self, in_channels, num_kernels, device="cuda"):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1).to(device)
        self.fc = nn.Linear(in_channels, num_kernels).to(device)
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def forward(self, x):
        x = self.global_avg_pool(x).to(self.device)
        x = x.view(x.size(0), -1).to(self.device)
        routing_coeffs = self.sigmoid(self.fc(x)).to(self.device)
        return routing_coeffs


class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, num_kernels=4, device="cuda"):
        super().__init__()
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = padding
        self.device = device
        self.kernels = nn.Parameter(
            torch.randn(num_kernels, in_channels, in_channels, kernel_size, kernel_size, device=device)
        )
        self.dynamic_routing = DynamicRoutingLayer(in_channels, num_kernels, device=device)

    def forward(self, x):
        routing_coeffs = self.dynamic_routing(x)
        routing_coeffs = F.softmax(routing_coeffs, dim=1)
        dynamic_kernels = torch.einsum("bk,klmij->blmij", routing_coeffs, self.kernels)
        outputs = []
        for i in range(x.size(0)):
            output_i = F.conv2d(
                x[i].unsqueeze(0).to(self.device),
                dynamic_kernels[i].to(self.device),
                stride=self.stride,
                padding=self.padding,
            )
            outputs.append(output_i)
        return torch.cat(outputs, dim=0)


class GConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filter_size=1,
        stride=1,
        padding=0,
        in_transformations=1,
        out_transformations=4,
        num_dynamic_kernels=4,
        device="cuda",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_transformations = in_transformations
        self.out_transformations = out_transformations
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.device = device

        total_in_channels = in_channels * in_transformations
        self.routing = DynamicRoutingLayer(total_in_channels, num_dynamic_kernels, device=device)
        self.kernel_bank = nn.Parameter(
            torch.empty(num_dynamic_kernels, out_channels, in_channels, filter_size, filter_size, device=device)
        )
        nn.init.kaiming_uniform_(self.kernel_bank, a=math.sqrt(5))

    def _rotate_kernels(self, kernels: torch.Tensor) -> torch.Tensor:
        rotations = [kernels]
        for k in range(1, self.out_transformations):
            rotations.append(torch.rot90(kernels, k, dims=(-2, -1)))
        return torch.stack(rotations, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        expected_channels = self.in_channels * self.in_transformations
        if channels != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} input channels (got {channels}); ensure the group dimension is folded into the channel axis."
            )

        routing_coeffs = self.routing(x)
        base_kernels = torch.einsum("bk,kocij->bocij", routing_coeffs, self.kernel_bank)
        rotated_kernels = self._rotate_kernels(base_kernels)

        x_grouped = x.view(batch_size, self.in_transformations, self.in_channels, height, width)

        outputs = []
        for b in range(batch_size):
            sample_outputs = []
            for r_out in range(self.out_transformations):
                conv_sum = None
                for r_in in range(self.in_transformations):
                    kernel_idx = (r_out - r_in) % self.out_transformations
                    conv_res = F.conv2d(
                        x_grouped[b, r_in].unsqueeze(0),
                        rotated_kernels[b, kernel_idx],
                        stride=self.stride,
                        padding=self.padding,
                    )
                    conv_sum = conv_res if conv_sum is None else conv_sum + conv_res
                sample_outputs.append(conv_sum)
            outputs.append(torch.cat(sample_outputs, dim=1))

        return torch.cat(outputs, dim=0)


class GMaxPool2d(nn.Module):
    def __init__(self, group="p4", device="cuda"):
        super().__init__()
        self.group = group
        self.device = device

    def forward(self, x):
        if self.group == "p4":
            x = nn.MaxPool2d(kernel_size=2)(x)
        return x.to(self.device)


def test_transform(transform_func):
    device = "cuda"
    x = torch.randn(128, 2, 1, 4096, requires_grad=True).to(device)
    g_conv = GConv2d(2, 32, filter_size=5, stride=1, padding=2).to(device)
    target = torch.zeros_like(g_conv(x))

    def forward_backward_pass():
        _ = transform_func(1, 2, 3)
        y = g_conv(x)
        loss = nn.MSELoss()(y, target)
        loss.backward()

    return forward_backward_pass


def measure_performance(transform_func):
    func = test_transform(transform_func)
    time_taken = timeit.timeit(func, number=1)
    mem_usage = memory_usage((func,), interval=0.1)
    max_mem_usage = max(mem_usage)
    return time_taken, max_mem_usage


def main():
    transform_methods = {
        "hilbert": group_element,
    }

    for name, method in transform_methods.items():
        print(f"\nTesting with {name} transform:")
        time_taken, max_mem_usage = measure_performance(method)
        print(f"Time taken: {time_taken:.6f} seconds")
        print(f"Max Memory Usage: {max_mem_usage:.2f} MiB")


if __name__ == "__main__":
    device = "cuda"
    print(device)
    x = torch.randn(128, 2, 1, 4096, requires_grad=True).to(device)

    g_conv = GConv2d(
        in_channels=2,
        out_channels=32,
        filter_size=5,
        stride=1,
        padding=2,
        in_transformations=1,
        out_transformations=4,
        device=device,
        num_dynamic_kernels=4,
    ).to(device)

    g_pool = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)).to(device)
    optimizer = torch.optim.Adam(g_conv.parameters(), lr=1e-4)

    start = time.time()
    y = g_conv(x)
    end = time.time()
    print(f"Forward time: {end - start} s")

    target = torch.zeros_like(y)
    loss = nn.MSELoss()(y, target)

    start = time.time()
    loss.backward()
    end = time.time()
    optimizer.step()
    print(f"Backward time: {end - start} s")

    main()
