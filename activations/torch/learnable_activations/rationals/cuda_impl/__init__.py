import os
import torch
import torch.nn as nn
from torch.autograd import Function

try:
    from .rational_cuda import rationals_forward, rationals_backward
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RationalsCUDAFunction(Function):
    @staticmethod
    def forward(ctx, x, coeff_numerator, coeff_denominator):
        if not CUDA_AVAILABLE:
            raise ImportError("CUDA implementation not available. Please install with CUDA support.")

        if not x.is_cuda:
            x = x.cuda()
        if not coeff_numerator.is_cuda:
            coeff_numerator = coeff_numerator.cuda()
        if not coeff_denominator.is_cuda:
            coeff_denominator = coeff_denominator.cuda()

        output = rationals_forward(x, coeff_numerator, coeff_denominator)
        ctx.save_for_backward(x, coeff_numerator, coeff_denominator)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not CUDA_AVAILABLE:
            raise ImportError("CUDA implementation not available. Please install with CUDA support.")

        x, coeff_numerator, coeff_denominator = ctx.saved_tensors
        grad_output = grad_output.to(dtype=torch.float32, device='cuda')
        grad_x, grad_coeff_numerator, grad_coeff_denominator = rationals_backward(
            grad_output.contiguous(),
            x.contiguous(),
            coeff_numerator.contiguous(),
            coeff_denominator.contiguous()
        )
        return grad_x, grad_coeff_numerator, grad_coeff_denominator

class RationalCUDA(nn.Module):
    def __init__(self, numerator_size: int = 5, denominator_size: int = 4, init: str = "normal", init_std: float = 1.0):
        super(RationalCUDA, self).__init__()
        if numerator_size > 16 or denominator_size > 16:
            raise ValueError("Coefficient sizes must be <= 16 due to internal limits.")

        self.numerator_size = numerator_size
        self.denominator_size = denominator_size
        self.init = init
        self.init_std = init_std

        if self.init == "normal":
            self.coeff_numerator = nn.Parameter(torch.randn((numerator_size,), dtype=torch.float32, device='cuda') * init_std)
            self.coeff_denominator = nn.Parameter(torch.randn((denominator_size,), dtype=torch.float32, device='cuda') * init_std)
        elif self.init == "uniform":
            self.coeff_numerator = nn.Parameter(torch.rand((numerator_size,), dtype=torch.float32, device='cuda'))
            self.coeff_denominator = nn.Parameter(torch.rand((denominator_size,), dtype=torch.float32, device='cuda'))
        else:
            raise ValueError(f"Unknown init type: {self.init}")

    def forward(self, x):
        device = x.device
        coeff_numerator = self.coeff_numerator.to(device)
        coeff_denominator = self.coeff_denominator.to(device)
        orig_dtype = x.dtype
        
        if orig_dtype == torch.float16:
            x = x.to(torch.float32)
            output = RationalsCUDAFunction.apply(x, coeff_numerator, coeff_denominator)
            output = output.to(orig_dtype)
        else:
            output = RationalsCUDAFunction.apply(x, coeff_numerator, coeff_denominator)
        return output

def init_rational_cuda_activation(numerator_size: int = 5, denominator_size: int = 4, init: str = "normal", init_std: float = 1.0) -> RationalCUDA:
    return RationalCUDA(numerator_size=numerator_size, denominator_size=denominator_size, init=init, init_std=init_std)

__all__ = ['RationalCUDA', 'RationalsCUDAFunction', 'CUDA_AVAILABLE', 'init_rational_cuda_activation']
