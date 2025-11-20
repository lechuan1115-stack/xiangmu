import itertools
import math
import time
import numpy as np
# import gcnn_cuda  # CUDA functions
# import gcnn_functions_cpp  # C++ functions
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal
import pywt
# from PyEMD import EMD
# import gcnn_functions_cpp  # C++ functions
import timeit
from memory_profiler import memory_usage
# import gcnn_cpu
# from gcnn_cpu import forward as gcnn_cpu_forward, backward as gcnn_cpu_backward
"""
    Converts a tuple (s, u, v) s - rotation, u - row, v - col, into a matrix group element
"""

# 尝试导入gcnn_cuda模块
try:
    import gcnn_cuda  # CUDA functions
except ImportError as e:
    print(f"Error importing gcnn_cuda: {e}")

# def group_element(s, u, v, m=0):
#     m_data = [
#         [
#             pow(-1, m) * math.cos(s * math.pi / 2),
#
#             pow(-1, m + 1) * math.sin(s * math.pi / 2),
#             u,
#         ],
#         [math.sin(s * math.pi / 2), math.cos(s * math.pi / 2), v],
#         [0.0, 0.0, 1.0],
#     ]
#     matrix = torch.tensor(m_data)
#     return matrix

# device = "cuda"
device = "cuda"

# 希尔伯特变换
def group_element(s, u, v, m=0):
    # print("s的大小：", s)
    # print("u的大小：", u)
    # print("v的大小：", v)

    # 使用scipy的希尔伯特变换
    hilbert_transformed = scipy.signal.hilbert([s, u, v])

    # 创建单位矩阵
    identity = np.eye(3)

    # 更新对角线的值
    for i in range(3):
        identity[i, i] += hilbert_transformed[i].real

    identity[0, 2] = u
    identity[1, 2] = v

    matrix = torch.tensor(identity, dtype=torch.float32).to('cuda')
    return matrix.to(device)

# 傅里叶变换
# def group_element(s, u, v, m=0):
#     # print("s的大小：", s)
#     # print("u的大小：", u)
#     # print("v的大小：", v)
#
#     # 使用scipy的傅里叶变换
#     fourier_transformed = scipy.fft.fft([s, u, v])
#
#     # 获取实部和虚部
#     real_part = fourier_transformed.real
#     imag_part = fourier_transformed.imag
#
#     # 创建单位矩阵
#     identity = np.eye(3)
#
#     # 更新对角线的值
#     for i in range(3):
#         identity[i, i] += real_part[i]
#
#     identity[0, 1] = imag_part[0]  # 或者根据需要更新其他位置
#     identity[1, 2] = imag_part[1]
#
#     identity[0, 2] = u
#     identity[1, 2] = v
#
#     matrix = torch.tensor(identity, dtype=torch.float32)
#     return matrix.to(device)



# 小波变换
# def group_element(s, u, v, m=0, wavelet='db1'):
#     # 确保输入是数组
#     s = np.array([s])
#     u = np.array([u])
#     v = np.array([v])
#
#     # 进行小波变换
#     coeffs_s = pywt.wavedec(s, wavelet)
#     coeffs_u = pywt.wavedec(u, wavelet)
#     coeffs_v = pywt.wavedec(v, wavelet)
#
#     # 取出小波变换的第一个系数（近似系数）
#     wavelet_transformed = [coeffs_s[0], coeffs_u[0], coeffs_v[0]]
#
#     # 创建单位矩阵
#     identity = np.eye(3)
#
#     # 更新对角线的值
#     for i in range(3):
#         identity[i, i] += wavelet_transformed[i].real
#
#     identity[0, 2] = u[0]
#     identity[1, 2] = v[0]
#
#     matrix = torch.tensor(identity, dtype=torch.float32)
#     return matrix.to(device)


# 短时傅里叶变换
# def group_element(s, u, v, m=0):
#     # 使用短时傅里叶变换
#     f, t, Zxx = scipy.signal.stft([s, u, v])
#
#     # 获取幅值
#     magnitude = np.abs(Zxx[:, 0])
#
#     # 确保 magnitude 的长度至少为 3
#     if len(magnitude) < 3:
#         magnitude = np.pad(magnitude, (0, 3 - len(magnitude)), 'constant')
#     else:
#         magnitude = magnitude[:3]
#
#     # 创建单位矩阵
#     identity = np.eye(3)
#
#     # 使用幅值更新对角线的值
#     for i in range(3):
#         identity[i, i] += magnitude[i]
#
#     identity[0, 2] = u
#     identity[1, 2] = v
#
#     matrix = torch.tensor(identity, dtype=torch.float32)
#     return matrix.to(device)

# 希尔伯特-黄变换
# def group_element(s, u, v, m=0):
#     # 将 s, u, v 组合成一个信号
#     signal = np.array([s, u, v])
#
#     # 使用 EMD 进行经验模态分解
#     emd = EMD()
#     imfs = emd(signal)
#
#     # 确保我们有足够的 IMFs
#     if imfs.shape[0] < 3:
#         imfs = np.pad(imfs, ((0, 3 - imfs.shape[0]), (0, 0)), 'constant')
#     else:
#         imfs = imfs[:3]
#
#     # 对每个 IMF 使用希尔伯特变换
#     hilbert_transformed = np.abs(scipy.signal.hilbert(imfs, axis=-1))
#
#     # 获取第一个希尔伯特变换的分量
#     hilbert_values = hilbert_transformed[:, 0]
#
#     # 创建单位矩阵
#     identity = np.eye(3)
#
#     # 更新对角线的值
#     for i in range(3):
#         identity[i, i] += hilbert_values[i]
#
#     identity[0, 2] = u
#     identity[1, 2] = v
#
#     matrix = torch.tensor(identity, dtype=torch.float32)
#     return matrix.to(device)



def group_element_inverse(matrix, rot=True):
    matrix_device = matrix.device  # 获取matrix的设备
    topleft = matrix[0][0]
    matrix += torch.eye(matrix.size(0)).to(matrix_device) * 1e-10
    # Topleft equals 0
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


# CUDA extension
class GConvFunctionsCUDA(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        filters,
        in_channels,
        out_channels,
        in_trans,
        out_trans,
        filter_size,
        stride,
        padding,
        ind1,
        ind2,
        ind3,
        device,
    ):


        ctx.in_channels = in_channels
        ctx.out_channels = out_channels
        ctx.in_trans = in_trans
        ctx.out_trans = out_trans
        ctx.filter_size = filter_size
        ctx.device = device
        ctx.stride = stride
        ctx.padding = padding



        filters_transformed = gcnn_cuda.forward(
            filters,
            in_channels,
            out_channels,
            in_trans,
            out_trans,
            filter_size,
            ind1,
            ind2,
            ind3,
        )


        ctx.save_for_backward(input, filters, filters_transformed, ind1, ind2, ind3)

        # print(f"Stride type: {type(ctx.stride)}, Stride value: {ctx.stride}")
        # print(f"Formatted stride: {stride}")

        output = F.conv2d(input, filters_transformed, stride=ctx.stride, padding=ctx.padding)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, filters, filters_transformed, ind1, ind2, ind3 = ctx.saved_tensors

        # print("CUDA backward")
        grad_input = torch.nn.grad.conv2d_input(
            input.shape,
            filters_transformed,
            grad_output,
            stride=ctx.stride,
            padding=ctx.padding,
        )
        grad_filters_trans = torch.nn.grad.conv2d_weight(
            input,
            filters_transformed.shape,
            grad_output,
            stride=ctx.stride,
            padding=ctx.padding,
        ).contiguous()


        grad_filters = gcnn_cuda.backward(
            ctx.out_channels,
            ctx.out_trans,
            ctx.in_channels,
            ctx.in_trans,
            ctx.filter_size,
            ind1,
            ind2,
            ind3,
            grad_filters_trans,
        )


        # 14 parameters in forward() so need to pad the number of fields returned
        return (
            grad_input,
            grad_filters,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )



# class GConvFunctionsCpp(torch.autograd.Function):
#     @staticmethod
#     def forward(
#         ctx,
#         input,
#         filters,
#         in_channels,
#         out_channels,
#         in_trans,
#         out_trans,
#         filter_size,
#         stride,
#         padding,
#         ind1,
#         ind2,
#         ind3,
#         device,
#     ):
#         ctx.in_channels = in_channels
#         ctx.out_channels = out_channels
#         ctx.in_trans = in_trans
#         ctx.out_trans = out_trans
#         ctx.filter_size = filter_size
#         ctx.device = device
#         ctx.stride = stride
#         ctx.padding = padding
#
#         filters_transformed = gcnn_functions_cpp.transform_filter(
#             out_channels,
#             out_trans,
#             in_channels,
#             in_trans,
#             filter_size,
#             ind1,
#             ind2,
#             ind3,
#             filters,
#         ).to(device)
#
#         # filters_transformed = torch.reshape(filters_transformed, (out_channels * out_trans, in_channels * in_trans, filter_size, filter_size)).to(device)
#         ctx.save_for_backward(input, filters, filters_transformed, ind1, ind2, ind3)
#         output = F.conv2d(
#             input, filters_transformed, stride=stride, padding=padding
#         ).to(device)
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, filters, filters_transformed, ind1, ind2, ind3 = ctx.saved_tensors
#
#         # Calculate dw = x * grad_output channel by channel
#         """
#         dw = torch.zeros_like(filters_transformed)
#         for i in range(dw.shape[1]): #in_channels
#             for j in range(dw.shape[0]): #out_channels
#                 #print(input[:, i, :, :].shape, grad_output[:, j, :, :].shape)
#                 dw[] += F.conv2d(input[:, i, :, :].unsqueeze(1), grad_output[:, j, :, :].unsqueeze(1))
#         print(dw.shape)
#         """
#         # print("C++ backward")
#
#         grad_input = torch.nn.grad.conv2d_input(
#             input.shape,
#             filters_transformed,
#             grad_output,
#             stride=ctx.stride,
#             padding=ctx.padding,
#         )
#         grad_filters_trans = torch.nn.grad.conv2d_weight(
#             input,
#             filters_transformed.shape,
#             grad_output,
#             stride=ctx.stride,
#             padding=ctx.padding,
#         )
#
#         grad_filters = gcnn_functions_cpp.gcnn_backward(
#             ctx.out_channels,
#             ctx.out_trans,
#             ctx.in_channels,
#             ctx.in_trans,
#             ctx.filter_size,
#             ind1,
#             ind2,
#             ind3,
#             grad_filters_trans,
#         ).to(ctx.device)
#
#         # 11 parameters in forward() so need to pad the number of fields returned
#         return (
#             grad_input,
#             grad_filters,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#         )


# # C++ extension
# class GConvFunctionsCpp(torch.autograd.Function):
#     @staticmethod
#     def forward(
#         ctx,
#         input,
#         filters,
#         in_channels,
#         out_channels,
#         in_trans,
#         out_trans,
#         filter_size,
#         stride,
#         padding,
#         ind1,
#         ind2,
#         ind3,
#         device,
#     ):
#         ctx.in_channels = in_channels
#         ctx.out_channels = out_channels
#         ctx.in_trans = in_trans
#         ctx.out_trans = out_trans
#         ctx.filter_size = filter_size
#         ctx.device = device
#         ctx.stride = stride
#         ctx.padding = padding
#
#         filters_transformed = gcnn_functions_cpp.transform_filter(
#             out_channels,
#             out_trans,
#             in_channels,
#             in_trans,
#             filter_size,
#             ind1,
#             ind2,
#             ind3,
#             filters,
#         ).to(device)
#
#         # filters_transformed = torch.reshape(filters_transformed, (out_channels * out_trans, in_channels * in_trans, filter_size, filter_size)).to(device)
#         ctx.save_for_backward(input, filters, filters_transformed, ind1, ind2, ind3)
#         output = F.conv2d(
#             input, filters_transformed, stride=stride, padding=padding
#         ).to(device)
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, filters, filters_transformed, ind1, ind2, ind3 = ctx.saved_tensors
#
#         # Calculate dw = x * grad_output channel by channel
#         """
#         dw = torch.zeros_like(filters_transformed)
#         for i in range(dw.shape[1]): #in_channels
#             for j in range(dw.shape[0]): #out_channels
#                 #print(input[:, i, :, :].shape, grad_output[:, j, :, :].shape)
#                 dw[] += F.conv2d(input[:, i, :, :].unsqueeze(1), grad_output[:, j, :, :].unsqueeze(1))
#         print(dw.shape)
#         """
#         # print("C++ backward")
#
#         grad_input = torch.nn.grad.conv2d_input(
#             input.shape,
#             filters_transformed,
#             grad_output,
#             stride=ctx.stride,
#             padding=ctx.padding,
#         )
#         grad_filters_trans = torch.nn.grad.conv2d_weight(
#             input,
#             filters_transformed.shape,
#             grad_output,
#             stride=ctx.stride,
#             padding=ctx.padding,
#         )
#
#         grad_filters = gcnn_functions_cpp.gcnn_backward(
#             ctx.out_channels,
#             ctx.out_trans,
#             ctx.in_channels,
#             ctx.in_trans,
#             ctx.filter_size,
#             ind1,
#             ind2,
#             ind3,
#             grad_filters_trans,
#         ).to(ctx.device)
#
#         # 11 parameters in forward() so need to pad the number of fields returned
#         return (
#             grad_input,
#             grad_filters,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#         )


# Custom forward and backward functions
# class GConvFunctions(torch.autograd.Function):
#     @staticmethod
#     def forward(
#         ctx,
#         input,
#         filters,
#         # dynamic_filters,
#         in_channels,
#         out_channels,
#         in_trans,
#         out_trans,
#         filter_size,
#         stride,
#         padding,
#         ind1,
#         ind2,
#         ind3,
#         device='cpu',
#     ):
#         # ctx.save_for_backward(input, filters)
#
#         # Save dimensions to ctx
#         ctx.in_channels = in_channels
#         ctx.out_channels = out_channels
#         ctx.in_trans = in_trans
#         ctx.out_trans = out_trans
#         ctx.filter_size = filter_size
#         ctx.device = device
#         ctx.stride = stride
#         ctx.padding = padding
#
#
#         filters_transformed = torch.zeros(
#             out_channels * out_trans,
#             in_channels * in_trans,
#             filter_size,
#             filter_size,
#             device=device,
#         )
#
#         # Can be optimized with CUDA
#         for i, s_prime, j, s, u, v in itertools.product(
#             range(out_channels),
#             range(out_trans),
#             range(in_channels),
#             range(in_trans),
#             range(filter_size),
#             range(filter_size),
#         ):
#
#             _s = ind1[s_prime, s, u, v].item()
#             _u = ind2[s_prime, s, u, v].item()
#             _v = ind3[s_prime, s, u, v].item()
#             filters_transformed[
#                 i * out_trans + s_prime, j * in_trans + s, u, v
#             ] = filters[i, j, _s, _u, _v]
#
#
#
#         ctx.save_for_backward(input, filters, filters_transformed, ind1, ind2, ind3)
#         return F.conv2d(input, filters_transformed, stride=stride, padding=padding)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, filters, filters_transformed, ind1, ind2, ind3 = ctx.saved_tensors
#
#         # Calculate dw = x * grad_output channel by channel
#         """
#         dw = torch.zeros_like(filters_transformed)
#         for i in range(dw.shape[1]): #in_channels
#             for j in range(dw.shape[0]): #out_channels
#                 #print(input[:, i, :, :].shape, grad_output[:, j, :, :].shape)
#                 dw[] += F.conv2d(input[:, i, :, :].unsqueeze(1), grad_output[:, j, :, :].unsqueeze(1))
#         print(dw.shape)
#         """
#
#         # grad_input = None
#         # grad_filters = torch.zeros_like(filters)
#
#         grad_input = torch.nn.grad.conv2d_input(
#             input.shape,
#             filters_transformed,
#             grad_output,
#             stride=ctx.stride,
#             padding=ctx.padding,
#         )
#         grad_filters_trans = torch.nn.grad.conv2d_weight(
#             input,
#             filters_transformed.shape,
#             grad_output,
#             stride=ctx.stride,
#             padding=ctx.padding,
#         ).contiguous()
#
#         # grad_filters_trans = torch.nn.grad.conv2d_weight(
#         #     input,
#         #     filters_transformed.shape,
#         #     grad_output,
#         #     stride=ctx.stride,
#         #     padding=ctx.padding,
#         # ).contiguous()
#
#         grad_filters = torch.zeros_like(filters).to(grad_output.device)
#         # print(grad_filters_trans.shape)
#         # grad_output.backward(filters)
#         for i, s_prime, j, s, u, v in itertools.product(
#             range(ctx.out_channels),
#             range(ctx.out_trans),
#             range(ctx.in_channels),
#             range(ctx.in_trans),
#             range(ctx.filter_size),
#             range(ctx.filter_size),
#         ):
#             # ref_prime = (s_prime > 3)
#             # ref = (s > 3)
#             # _s, _u, _v = group_element_inverse(torch.inverse(group_element(s_prime, 0, 0, m=ref_prime)) * group_element(s, u, v, m=ref))
#
#
#             _s = ind1[s_prime, s, u, v].item()
#             _u = ind2[s_prime, s, u, v].item()
#             _v = ind3[s_prime, s, u, v].item()
#             # Update grad_filters according to grad_output
#             grad_filters[i, j, _s, _u, _v] += grad_filters_trans[
#                 i * ctx.out_trans + s_prime, j * ctx.in_trans + s, u, v
#             ]
#
#
#         # 11 parameters in forward() so need to pad the number of fields returned
#         return (
#             grad_input,
#             grad_filters,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#         )




# 有动态卷积核

class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, num_kernels=4, device = 'cuda'):
        super(DynamicConv2d, self).__init__()

        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = padding
        self.device = device

        # 多个候选卷积核: [K, C_out(=in_channels), C_in, k, k]
        self.kernels = nn.Parameter(
            torch.randn(num_kernels, in_channels, in_channels, kernel_size, kernel_size, device=device)
        )

        # 动态路由层，输出每个kernel的系数
        self.dynamic_routing = DynamicRoutingLayer(in_channels, num_kernels, device=device)

    def forward(self, x):
        # [B, K]
        routing_coeffs = self.dynamic_routing(x)
        routing_coeffs = F.softmax(routing_coeffs, dim=1)

        # [B, C_out, C_in, k, k]
        dynamic_kernels = torch.einsum('bk,klmij->blmij', routing_coeffs, self.kernels)

        outputs = []
        for i in range(x.size(0)):
            # 对每个样本用自己的一组卷积核
            output_i = F.conv2d(
                x[i].unsqueeze(0).to(self.device),
                dynamic_kernels[i].to(self.device),
                stride=self.stride,
                padding=self.padding
            )
            outputs.append(output_i)
        output = torch.cat(outputs, dim=0)
        return output


class DynamicRoutingLayer(nn.Module):
    def __init__(self, in_channels, num_kernels, device="cuda"):
        super(DynamicRoutingLayer, self).__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1).to(device)
        self.fc = nn.Linear(in_channels, num_kernels).to(device)
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def forward(self, x):
        x = self.global_avg_pool(x).to(self.device)     # [B, C, 1, 1]
        x = x.view(x.size(0), -1).to(self.device)       # [B, C]
        routing_coeffs = self.sigmoid(self.fc(x)).to(self.device)  # [B, K]
        return routing_coeffs

# # 没有动态卷积核
# class GConv2d(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         filter_size=1,
#         stride=1,
#         padding=0,
#         in_transformations=1,
#         out_transformations=4,
#         group="p4",
#         device="cuda",
#     ):
#         super().__init__()
#         # Filters stored as a K_l x K_{l-1} x S_{l_1} x n x n
#         self.filters = nn.Parameter(
#             torch.zeros(
#                 out_channels,
#                 in_channels,
#                 in_transformations,
#                 filter_size,
#                 filter_size,
#                 device=device,
#             )
#         )
#         nn.init.kaiming_uniform_(self.filters, mode="fan_in", nonlinearity="relu")
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.in_trans = in_transformations
#         self.out_trans = out_transformations
#         self.filter_size = filter_size
#         self.device = device
#
#         self.stride = stride
#         self.padding = padding
#
#         # Filter transformations stored as a K_l x S_l x K_{l-1} x S_{l_1} x n x n
#         # Precompute lookup indices
#         self.ind1 = torch.zeros(
#             size=(self.out_trans, self.in_trans, self.filter_size, self.filter_size),
#             dtype=torch.int32,
#         ).to(device)
#         self.ind2 = torch.zeros(
#             size=(self.out_trans, self.in_trans, self.filter_size, self.filter_size),
#             dtype=torch.int32,
#         ).to(device)
#         self.ind3 = torch.zeros(
#             size=(self.out_trans, self.in_trans, self.filter_size, self.filter_size),
#             dtype=torch.int32,
#         ).to(device)
#
#         half_x = self.filter_size / 2.0 - 0.5
#         half_y = self.filter_size / 2.0 - 0.5
#         for s_prime, s, u, v in itertools.product(
#             range(self.out_trans),
#             range(self.in_trans),
#             range(self.filter_size),
#             range(self.filter_size),
#         ):
#             ref_prime = s_prime > 3
#             ref = s > 3
#             # Calculate new indices
#             new_coords = group_element_inverse(
#                 torch.mm(
#                     torch.inverse(group_element(s_prime, 0, 0, m=ref_prime)),
#                     # group_element(s, u - half_x, -1 * (v - half_y), m=ref),
#                     group_element(s, v - half_x, -1 * (u - half_y), m=ref),
#                 ),
#                 rot=(self.in_trans > 1),
#             )
#             _s = new_coords[0]
#             _u = round(-1 * new_coords[2] + half_y)
#             _v = round(new_coords[1] + half_x)
#
#             self.ind1[s_prime, s, u, v] = _s
#             self.ind2[s_prime, s, u, v] = _u
#             self.ind3[s_prime, s, u, v] = _v
#
#     def forward(self, x):
#         # if self.device == "cpu":
#         #     func = GConvFunctionsCpp
#         # else:
#         func = GConvFunctionsCUDA
#         return func.apply(
#             x,
#             self.filters,
#             self.in_channels,
#             self.out_channels,
#             self.in_trans,
#             self.out_trans,
#             self.filter_size,
#             self.stride,
#             self.padding,
#             self.ind1,
#             self.ind2,
#             self.ind3,
#             self.device,
#         )
#


class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, num_kernels=4, device = 'cuda'):
        super(DynamicConv2d, self).__init__()

        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = padding

        self.device = device

        # 初始化多个卷积核，输出通道数为2
        # self.kernels = nn.Parameter(torch.randn(num_kernels, 2, in_channels, kernel_size, kernel_size))
        # self.kernels = nn.Parameter(torch.randn(num_kernels, in_channels, in_channels, kernel_size, kernel_size))
        self.kernels = nn.Parameter(torch.randn(num_kernels, in_channels, in_channels, kernel_size, kernel_size, device=device))

        # 动态路由层
        self.dynamic_routing = DynamicRoutingLayer(in_channels, num_kernels, device=device)

    def forward(self, x):
        # Get the routing coefficients: Shape [batch_size, num_kernels]
        routing_coeffs = self.dynamic_routing(x)
        routing_coeffs = F.softmax(routing_coeffs, dim=1)

        # 使用路由系数结合卷积核
        dynamic_kernels = torch.einsum('bk,klmij->blmij', routing_coeffs, self.kernels)

        outputs = []
        for i in range(x.size(0)):
            output_i = F.conv2d(x[i].unsqueeze(0).to(self.device), dynamic_kernels[i].to(self.device), stride=self.stride, padding=self.padding)
            outputs.append(output_i)
        output = torch.cat(outputs, dim=0)

        return output


class DynamicRoutingLayer(nn.Module):
    def __init__(self, in_channels, num_kernels, device="cuda"):
        super(DynamicRoutingLayer, self).__init__()

        # 全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1).to(device)

        # 一个全连接层计算路由系数
        self.fc = nn.Linear(in_channels, num_kernels).to(device)

        self.device = device

        # 使用sigmoid函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 使用全局平均池化得到特征图的全局信息
        x = self.global_avg_pool(x).to(self.device)
        x = x.view(x.size(0), -1).to(self.device)

        # print(x.shape)
        # 计算路由系数
        routing_coeffs = self.sigmoid(self.fc(x)).to(self.device)

        return routing_coeffs


"""
Here we implement coset pooling over H and subsample over cosets gH, where
H = The 4 rotations around the origin if G = p4
"""


class GMaxPool2d(nn.Module):
    def __init__(self, group="p4", device="cuda"):
        super().__init__()
        self.group = group
        self.device = device

    def forward(self, x):
        # x is of shape [B, channels x |H|, width, height]
        # out should be of shape [B, channels, width, height]
        if self.group == "p4":
            """
            if self.device == "cpu":
                out = gcnn_functions_cpp.gmaxpool_forward(x)
                out.requires_grad_(True)
            else:
                out = gcnn_cuda.gmaxpool_forward(x)
                out.requires_grad_(True)
            """
            # x = F.reshape(x, (xs[0], xs[1] * xs[2], xs[3], xs[4]))
            # x = F.max_pooling_2d(x, ksize, stride, pad, cover_all, use_cudnn)
            # x = F.reshape(x, (xs[0], xs[1], xs[2], x.data.shape[2], x.data.shape[3]))
            # xs = x.shape
            # x = torch.reshape(x, (xs[0], xs[1] * xs[2], xs[3], xs[4]))
            """
            out = torch.zeros(
                x.shape[0], int(x.shape[1] / 4), x.shape[2], x.shape[3]
            )
            for b, i, u, v in itertools.product(
                range(x.shape[0]),
                range(out.shape[0]),
                range(x.shape[2]),
                range(x.shape[3]),
            ):
                out[b, i, u, v] = max(
                    x[b, i, u, v],
                    x[b, i + out.shape[0], u, v],
                    x[b, i + out.shape[0] * 2, u, v],
                    x[b, i + out.shape[0] * 3, u, v],
                )
            """
            x = nn.MaxPool2d(kernel_size=2)(x)

        return x.to(self.device)


def test_transform(transform_func):
    device = "cuda"
    x = torch.randn(128, 2, 1, 4096, requires_grad=True).to(device)
    g_conv = GConv2d(2, 32, filter_size=5, stride=1, padding=2).to(device)
    target = torch.zeros_like(g_conv(x))

    def forward_backward_pass():
        s, u, v = 1, 2, 3
        transform_result = transform_func(s, u, v)  # 执行变换
        y = g_conv(x)
        loss = nn.MSELoss()(y, target)
        loss.backward()

    return forward_backward_pass


def measure_performance(transform_func):
    func = test_transform(transform_func)

    # 测量前向和后向传递的时间
    time_taken = timeit.timeit(func, number=1)

    # 测量内存使用
    mem_usage = memory_usage((func,), interval=0.1)
    max_mem_usage = max(mem_usage)

    return time_taken, max_mem_usage


def main():
    transform_methods = {
        'hilbert': group_element,
        # 'fourier': group_element,
        # 'wavelet': group_element,
        # 'WVD': group_element,
        # 'HHT': group_element
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

    # 使用带动态卷积核的 GConv2d
    g_conv = GConv2d(
        in_channels=2,
        out_channels=32,
        filter_size=5,
        stride=1,
        padding=2,
        in_transformations=1,
        out_transformations=4,
        device=device,
        num_dynamic_kernels=4,   # 动态卷积核个数
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

