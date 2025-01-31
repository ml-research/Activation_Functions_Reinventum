import torch
import os
from torch.utils.cpp_extension import load_inline
from torch.autograd import Function
import torch.nn as nn

# Set a custom directory for PyTorch extensions so it's importable and doesn't clutter the root.
# Often unnecessary on servers like DGX machines, but useful in other environments.
os.environ['TORCH_EXTENSIONS_DIR'] = './torch_extensions'

"""
====================================================================
CUDA-BASED RATIONAL ACTIVATION (FORWARD + BACKWARD) - EXPLANATION
====================================================================

1. Shared Memory:
   We load the numerator and denominator coefficients (e.g., 5 for numerator, 4 for denominator)
   into shared memory to minimize global memory reads. Since these coefficients are the same for
   every element, caching them in shared memory significantly speeds up performance.

2. Horner's Method:
   An efficient way to evaluate polynomials:
     P(x) = a_0 + a_1 x + a_2 x^2 + ...
   rewrites as
     P(x) = a_0 + x (a_1 + x (a_2 + ... ) )
   This avoids repeatedly computing powers of x and can offer better numerical stability.

3. Absolute Value + 1 in Denominator:
   We keep the denominator away from zero by doing ( |Q_tilde| + 1 ).
   This matches the logic in the original PyTorch-based activation. The backward pass includes
   the sign of Q_tilde to handle d/dx(|x|).

4. Atomic Adds:
   When many threads compute partial derivatives with respect to the same coefficient(s),
   we need an atomic mechanism to sum them safely (thread-safe).

5. Grid-Stride Loop:
   A common CUDA pattern to handle large input sizes. Each thread processes elements:
       idx, idx+stride, idx+2*stride, ...
   until we cover all inputs.

6. Backward Pass Logic:
   - Recompute numerator and denominator polynomials (and their derivatives) using Horner’s method.
   - Use chain rule + quotient rule to get gradients wrt x, numerator coefficients, and denominator coefficients.
   - Reduce within-block partial sums, then use atomicAdd to produce the final global sums.
"""

# -----------------------------------------------------------------------------------------
#                                  CUDA KERNEL CODE
# -----------------------------------------------------------------------------------------
cuda_code = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>
#include <ATen/AccumulateType.h>

// The numerator has 5 coefficients (M = 5), the denominator has 4 coefficients (N = 4).
// Adjust these if you have a different polynomial structure.
#define M 5
#define N 4

// ---------------------------------------------------------------------
//  Helper Functions + Forward Declarations
// ---------------------------------------------------------------------

// fused_mul_add:
// Encourages the compiler to use fused multiply-add instructions (FMA), 
// which combine multiplication and addition for speed and possibly more accuracy.
template <typename scalar_t>
__device__ __forceinline__ scalar_t fused_mul_add(scalar_t a, scalar_t b, scalar_t c) {
    return a * b + c;
}

// Overload for half precision (at::Half):
template <>
__device__ __forceinline__ at::Half fused_mul_add(at::Half a, at::Half b, at::Half c) {
    // cast to float because half * half is not directly supported at lower GPU arch levels
    return at::Half(float(a) * float(b) + float(c));
}

// abs_val:
// A templated absolute value function so we can handle both float/double and half (at::Half).
template <typename scalar_t>
__device__ __forceinline__ scalar_t abs_val(scalar_t x) {
    return fabs(x);
}

// Overload for half
template <>
__device__ __forceinline__ at::Half abs_val(at::Half x) {
    return at::Half(fabs(float(x)));
}

// ---------------------------------------------------------------------
//  AtomicAdd Wrappers
// ---------------------------------------------------------------------
// We need atomicAdd because multiple threads can update the same gradient location simultaneously.
//
// We specialize for float, double, and at::Half. Although recent CUDA versions have built-in
// half-precision atomicAdd on some architectures, this code ensures broader compatibility.
// Half not working producing NaNs -> therefore in init always cast input >= 32 precisison for activation

template <typename T>
__device__ __forceinline__ void atomicAddWrapper(T* address, T val);

// float specialization
template <>
__device__ __forceinline__ void atomicAddWrapper<float>(float* address, float val) {
    atomicAdd(address, val);
}

// double specialization
template <>
__device__ __forceinline__ void atomicAddWrapper<double>(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    atomicAdd(address, val);
#else
    // For older architectures that do not support atomicAdd(double) natively,
    // we implement a CAS-based approach.
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
#endif
}

// half (at::Half) specialization
template <>
__device__ __forceinline__ void atomicAddWrapper<at::Half>(at::Half* address, at::Half val) {
    // cast to char* to fix potential misalignment
    unsigned int* address_as_ui = (unsigned int*)(((char*)address) - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    do {
        assumed = old;
        
        union {
            at::Half a;
            __half h;
        } u;
        u.a = val;
        __half_raw raw_val = __half_raw(u.h);
        
        // Re-insert the new value into the correct half of the 32-bit word
        unsigned int new_val;
        if (((size_t)address & 2)) {
            new_val = (old & 0x0000FFFF) | (raw_val.x << 16);
        } else {
            new_val = (old & 0xFFFF0000) | raw_val.x;
        }
        old = atomicCAS(address_as_ui, assumed, new_val);
    } while (assumed != old);
}

// ---------------------------------------------------------------------
//  Forward Kernel
// ---------------------------------------------------------------------
//
// This kernel computes output[idx] = numerator(x_val) / ( |denominator(x_val)| + 1 )
// for each x_val in the input array x.
//
// Key techniques:
// - Shared Memory to hold polynomial coefficients for reuse within a block.
// - Horner's method to efficiently compute polynomials with minimal multiplications.
// - A grid-stride loop to handle large input sizes.

template <typename scalar_t>
__global__ void rationals_cuda_inline_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ coeff_num,
    const scalar_t* __restrict__ coeff_den,
    scalar_t* __restrict__ output,
    int num_elements) {

    // 1. Shared memory for polynomial coefficients
    __shared__ scalar_t shared_coeff_num[M];
    __shared__ scalar_t shared_coeff_den[N];

    // 2. Load coefficients in reverse order for convenient Horner's method
    //    Only threadIdx.x < M/N will copy, others skip.
    if (threadIdx.x < M) {
        shared_coeff_num[threadIdx.x] = coeff_num[M - 1 - threadIdx.x];
    }
    if (threadIdx.x < N) {
        shared_coeff_den[threadIdx.x] = coeff_den[N - 1 - threadIdx.x];
    }
    __syncthreads();

    // 3. Grid-stride loop setup
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 4. Evaluate the polynomial for all elements this thread is responsible for
    for (; idx < num_elements; idx += stride) {
        scalar_t x_val = x[idx];

        // Horner's method for numerator
        scalar_t numerator = shared_coeff_num[0];
        #pragma unroll
        for (int j = 1; j < M; ++j) {
            numerator = fused_mul_add(numerator, x_val, shared_coeff_num[j]);
        }

        // Horner's method for denominator
        scalar_t denominator = shared_coeff_den[0];
        #pragma unroll
        for (int j = 1; j < N; ++j) {
            denominator = fused_mul_add(denominator, x_val, shared_coeff_den[j]);
        }

        // Multiply the denominator polynomial by x_val (as the original PyTorch code effectively had one extra power of x).
        denominator = denominator * x_val;

        // Add abs(...) + 1 to avoid divide-by-zero or negative issues
        scalar_t denom = abs_val(denominator) + scalar_t(1.0);

        // Final output
        output[idx] = numerator / denom;
    }
}

// ---------------------------------------------------------------------
//  Backward Kernel
// ---------------------------------------------------------------------
//
// We compute partial derivatives w.r.t. x, numerator coefficients, and denominator coefficients.
// Each thread accumulates local sums of gradient contributions, then reduces them within the block,
// and finally uses atomicAdd to combine them into the global gradient arrays.
//
// Steps:
// 1) Load the same polynomial coefficients into shared memory (reverse order).
// 2) Each thread does grid-stride loop over input elements.
// 3) Compute forward intermediate results (numerator, denominator) + their derivatives (Horner's).
// 4) Use chain rule + quotient rule to get grad wrt x, numerator coefficients, denominator coefficients.
// 5) Block-level reduction for partial sums, then atomicAdd to global arrays.

template <typename scalar_t>
__global__ void rationals_cuda_inline_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ coeff_num,
    const scalar_t* __restrict__ coeff_den,
    scalar_t* __restrict__ grad_x,
    scalar_t* __restrict__ grad_coeff_num,
    scalar_t* __restrict__ grad_coeff_den,
    int num_elements) {

    // Type used for intermediate sums (can be float even if scalar_t is half).
    using accscalar_t = at::acc_type<scalar_t, true>;

    // 1. Load coefficients into shared memory
    __shared__ scalar_t shared_coeff_num[M];
    __shared__ scalar_t shared_coeff_den[N];

    if (threadIdx.x < M) {
        shared_coeff_num[threadIdx.x] = coeff_num[M - 1 - threadIdx.x];
    }
    if (threadIdx.x < N) {
        shared_coeff_den[threadIdx.x] = coeff_den[N - 1 - threadIdx.x];
    }
    __syncthreads();

    // 2. Initialize local accumulators for partial gradients
    accscalar_t local_grad_coeff_num[M] = {0};
    accscalar_t local_grad_coeff_den[N] = {0};

    // 3. Thread index and stride for grid-stride loop
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 4. Loop over all elements this thread is responsible for
    for (; idx < num_elements; idx += stride) {
        scalar_t x_val    = x[idx];
        scalar_t grad_out = grad_output[idx];  // gradient from next layer

        // Recompute numerator polynomial & its derivative wrt x
        scalar_t numerator = shared_coeff_num[0];
        scalar_t numerator_derivative = scalar_t(0);
        #pragma unroll
        for (int i = 1; i < M; ++i) {
            // derivative of Horner’s method
            numerator_derivative = fused_mul_add(numerator_derivative, x_val, numerator);
            numerator = fused_mul_add(numerator, x_val, shared_coeff_num[i]);
        }

        // Recompute denominator polynomial Q and Q's derivative
        scalar_t Q      = shared_coeff_den[0];
        scalar_t Q_prime = scalar_t(0);
        #pragma unroll
        for (int i = 1; i < N; ++i) {
            Q_prime = fused_mul_add(Q_prime, x_val, Q);
            Q       = fused_mul_add(Q, x_val, shared_coeff_den[i]);
        }

        // Multiply by x to get Q_tilde; also compute derivative Q_tilde'
        scalar_t Q_tilde        = Q * x_val;
        scalar_t Q_tilde_prime  = Q_prime * x_val + Q;

        // denom = |Q_tilde| + 1
        scalar_t denom = abs_val(Q_tilde) + scalar_t(1.0);

        // sign(Q_tilde) for derivative of abs(Q_tilde)
        scalar_t sign_Q_tilde = (Q_tilde >= scalar_t(0)) ? scalar_t(1.0) : scalar_t(-1.0);
        scalar_t denom_derivative = sign_Q_tilde * Q_tilde_prime;

        // 5. Gradient wrt x:
        //    output = numerator / denom
        //    => d/dx = (numerator' * denom - numerator * denom') / denom^2
        scalar_t denom_squared = denom * denom;
        scalar_t grad_input = (numerator_derivative * denom - numerator * denom_derivative) / denom_squared;
        grad_x[idx] = grad_out * grad_input;

        // 6. Gradients wrt numerator coefficients and denominator coefficients
        //    common_factor_num = (grad_out / denom)
        //    common_factor_den = -grad_out * numerator * sign_Q_tilde / denom^2
        scalar_t inv_denom = scalar_t(1.0) / denom;
        scalar_t inv_denom_squared = inv_denom * inv_denom;

        scalar_t common_factor_num = grad_out * inv_denom;
        scalar_t common_factor_den = -grad_out * numerator * sign_Q_tilde * inv_denom_squared;

        // Horner’s method derivative w.r.t. each coefficient: each coefficient in
        // a polynomial a_0 + a_1 x + ... basically sees partial derivative = x^i for the i-th coefficient.
        // We accumulate those partial derivatives in local arrays.

        // Numerator coefficients
        scalar_t accum = scalar_t(1.0);
        #pragma unroll
        for (int i = 0; i < M; ++i) {
            local_grad_coeff_num[i] += accscalar_t(accum * common_factor_num);
            accum *= x_val;
        }

        // Denominator coefficients
        accum = x_val;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            local_grad_coeff_den[i] += accscalar_t(accum * common_factor_den);
            accum *= x_val;
        }
    }

    // 7. Now we reduce the local partial sums within the block using shared memory.
    //    We'll place them in shared memory, do a standard block reduction, then 
    //    atomicAdd to the global gradient arrays.

    extern __shared__ char shared_mem_raw[];
    accscalar_t* shared_grad = reinterpret_cast<accscalar_t*>(shared_mem_raw);

    // Layout: first M * blockDim.x for numerator, next N * blockDim.x for denominator
    accscalar_t* shared_grad_num = shared_grad;
    accscalar_t* shared_grad_den = shared_grad + M * blockDim.x;

    // Copy local sums into shared memory
    for (int i = 0; i < M; ++i) {
        shared_grad_num[i * blockDim.x + threadIdx.x] = local_grad_coeff_num[i];
    }
    for (int i = 0; i < N; ++i) {
        shared_grad_den[i * blockDim.x + threadIdx.x] = local_grad_coeff_den[i];
    }
    __syncthreads();

    // 8. Reduce within the block to get a single sum per coefficient
    for (int i = 0; i < M; ++i) {
        accscalar_t* sdata = &shared_grad_num[i * blockDim.x];
        unsigned int tid = threadIdx.x;

        // Parallel reduction
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        // The final block sum is in sdata[0]; add it to global using atomicAdd
        if (tid == 0) {
            atomicAddWrapper(&grad_coeff_num[i], scalar_t(sdata[0]));
        }
    }

    for (int i = 0; i < N; ++i) {
        accscalar_t* sdata = &shared_grad_den[i * blockDim.x];
        unsigned int tid = threadIdx.x;

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            atomicAddWrapper(&grad_coeff_den[i], scalar_t(sdata[0]));
        }
    }
}

// ---------------------------------------------------------------------
//  Host-Side Entry Points for Forward/Backward
// ---------------------------------------------------------------------

// Launch function for forward pass
void rationals_cuda_inline_forward(
    at::Tensor x,
    at::Tensor coeff_numerator,
    at::Tensor coeff_denominator,
    at::Tensor output) {

    // Make sure all tensors are contiguous in memory
    x = x.contiguous();
    coeff_numerator = coeff_numerator.contiguous();
    coeff_denominator = coeff_denominator.contiguous();
    output = output.contiguous();

    // Flatten the input/output for simpler 1D kernel launches
    auto x_flat = x.view(-1);
    auto output_flat = output.view(-1);
    int num_elements = x_flat.size(0);

    // Set typical block and grid sizes. Adjust as needed.
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    // Dispatch the kernel for float, double, or half
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rationals_cuda_inline_forward", ([&] {
        rationals_cuda_inline_forward_kernel<scalar_t><<<blocks, threads>>>(
            x_flat.data_ptr<scalar_t>(),
            coeff_numerator.data_ptr<scalar_t>(),
            coeff_denominator.data_ptr<scalar_t>(),
            output_flat.data_ptr<scalar_t>(),
            num_elements);
    }));
}

// Launch function for backward pass
void rationals_cuda_inline_backward(
    at::Tensor grad_output,
    at::Tensor x,
    at::Tensor coeff_numerator,
    at::Tensor coeff_denominator,
    at::Tensor grad_x,
    at::Tensor grad_coeff_numerator,
    at::Tensor grad_coeff_denominator) {

    // Also ensure all are contiguous
    grad_output = grad_output.contiguous();
    x = x.contiguous();
    coeff_numerator = coeff_numerator.contiguous();
    coeff_denominator = coeff_denominator.contiguous();
    grad_x = grad_x.contiguous();
    grad_coeff_numerator = grad_coeff_numerator.contiguous();
    grad_coeff_denominator = grad_coeff_denominator.contiguous();

    // Flatten
    auto x_flat = x.view(-1);
    auto grad_output_flat = grad_output.view(-1);
    auto grad_x_flat = grad_x.view(-1);
    int num_elements = x_flat.size(0);

    // Launch configuration
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    // We need shared memory to hold partial sums: (M+N)*threads*sizeof(accscalar_t)
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rationals_cuda_inline_backward", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        size_t shared_mem_size = (M + N) * threads * sizeof(accscalar_t);

        rationals_cuda_inline_backward_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            grad_output_flat.data_ptr<scalar_t>(),
            x_flat.data_ptr<scalar_t>(),
            coeff_numerator.data_ptr<scalar_t>(),
            coeff_denominator.data_ptr<scalar_t>(),
            grad_x_flat.data_ptr<scalar_t>(),
            grad_coeff_numerator.data_ptr<scalar_t>(),
            grad_coeff_denominator.data_ptr<scalar_t>(),
            num_elements);
    }));
}
'''.strip()

# -----------------------------------------------------------------------------------------
#                                  C++ WRAPPER CODE
# -----------------------------------------------------------------------------------------
cpp_code = r'''
#include <torch/extension.h>
#include <vector>

// Declarations of the CUDA launch functions (defined above)
void rationals_cuda_inline_forward(
    at::Tensor x,
    at::Tensor coeff_numerator,
    at::Tensor coeff_denominator,
    at::Tensor output);

void rationals_cuda_inline_backward(
    at::Tensor grad_output,
    at::Tensor x,
    at::Tensor coeff_numerator,
    at::Tensor coeff_denominator,
    at::Tensor grad_x,
    at::Tensor grad_coeff_numerator,
    at::Tensor grad_coeff_denominator);

// ---------------------------------------------------------------------
//  PUBLIC INTERFACES: Called from PyTorch
// ---------------------------------------------------------------------

// Forward interface: returns output = activation(x)
at::Tensor rationals_cuda_forward(
    at::Tensor x,
    at::Tensor coeff_numerator,
    at::Tensor coeff_denominator) {

    auto output = at::zeros_like(x);
    rationals_cuda_inline_forward(x, coeff_numerator, coeff_denominator, output);
    return output;
}

// Backward interface: returns a vector of three tensors:
// [ grad_x, grad_coeff_numerator, grad_coeff_denominator ]
std::vector<at::Tensor> rationals_cuda_backward(
    at::Tensor grad_output,
    at::Tensor x,
    at::Tensor coeff_numerator,
    at::Tensor coeff_denominator) {

    auto grad_x = at::zeros_like(x);

    // Keep gradients for coeff_numerator and coeff_denominator in float 
    // (or double) for higher precision, if desired. This code uses float.
    auto grad_coeff_numerator = at::zeros_like(coeff_numerator, coeff_numerator.options().dtype(at::kFloat));
    auto grad_coeff_denominator = at::zeros_like(coeff_denominator, coeff_denominator.options().dtype(at::kFloat));

    rationals_cuda_inline_backward(
        grad_output,
        x,
        coeff_numerator,
        coeff_denominator,
        grad_x,
        grad_coeff_numerator,
        grad_coeff_denominator);

    return {grad_x, grad_coeff_numerator, grad_coeff_denominator};
}
'''.strip()

# -----------------------------------------------------------------------------------------
#                      LOAD THE EXTENSION (JIT COMPILE)
# -----------------------------------------------------------------------------------------
try:
    rationals_cuda_ext = load_inline(
        name='rationals_cuda_ext',
        cpp_sources=cpp_code,
        cuda_sources=cuda_code,
        functions=['rationals_cuda_forward', 'rationals_cuda_backward'],
        verbose=True,
        extra_cuda_cflags=['-O2', '--use_fast_math']
    )
except Exception as e:
    print(f"Failed to compile CUDA extension: {str(e)}")
    raise


###############################################################################
#                  Init Rationals CUDA Function & Activation                  #
# - casting to 32 always
###############################################################################
class RationalsCUDAFunction(Function):
    """
    Custom autograd Function that uses compiled CUDA kernels to perform
    forward/backward passes of a rational activation.
    """
    @staticmethod
    def forward(ctx, x, coeff_numerator, coeff_denominator):
        # Ensure all inputs are on CUDA
        if not x.is_cuda:
            x = x.cuda()
        if not coeff_numerator.is_cuda:
            coeff_numerator = coeff_numerator.cuda()
        if not coeff_denominator.is_cuda:
            coeff_denominator = coeff_denominator.cuda()

        # Call the forward kernel from the compiled extension
        outputs = rationals_cuda_ext.rationals_cuda_forward(x, coeff_numerator, coeff_denominator)

        # Save tensors for backward
        ctx.save_for_backward(x, coeff_numerator, coeff_denominator)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        x, coeff_numerator, coeff_denominator = ctx.saved_tensors
        grad_output = grad_output.to(dtype=torch.float32, device='cuda')

        # Call the backward kernel from the compiled extension
        grad_x, grad_coeff_numerator, grad_coeff_denominator = rationals_cuda_ext.rationals_cuda_backward(
            grad_output.contiguous(),
            x.contiguous(),
            coeff_numerator.contiguous(),
            coeff_denominator.contiguous()
        )
        return grad_x, grad_coeff_numerator, grad_coeff_denominator


class RationalsCUDAActivation(nn.Module):
    """
    PyTorch nn.Module wrapper around the custom CUDA rational activation.
    Allows specifying:
      - numerator_size: number of numerator coefficients
      - denominator_size: number of denominator coefficients
      - init: type of initialization ("normal", "uniform", etc.)
    """
    def __init__(self,
                 numerator_size: int = 5,
                 denominator_size: int = 4,
                 init: str = "normal",
                 init_std: float = 1.0):
        """
        Args:
            numerator_size (int): Number of coefficients in the numerator polynomial.
            denominator_size (int): Number of coefficients in the denominator polynomial.
            init (str): Initialization type, e.g. "normal" or "uniform".
            init_std (float): Standard deviation if using 'normal' init.
        """
        super(RationalsCUDAActivation, self).__init__()

        self.numerator_size = numerator_size
        self.denominator_size = denominator_size
        self.init = init
        self.init_std = init_std

        # Initialize coefficients
        if self.init == "normal":
            self.coeff_numerator = nn.Parameter(
                torch.randn((numerator_size,), dtype=torch.float32)
            )
            self.coeff_denominator = nn.Parameter(
                torch.randn((denominator_size,), dtype=torch.float32)
            )
        elif self.init == "uniform":
            self.coeff_numerator = nn.Parameter(
                torch.normal(mean=0.0, std=1.0, size=(numerator_size,), dtype=torch.float32)
            )
            self.coeff_denominator = nn.Parameter(
                torch.normal(mean=0.0, std=1.0, size=(denominator_size,), dtype=torch.float32)
            )
        else:
            raise ValueError(f"Unknown init type: {self.init}")

    def forward(self, x):
        """
        Applies the rational activation to input x using
        the custom CUDA kernels.
        """
        device = x.device
        coeff_numerator = self.coeff_numerator.to(device)
        coeff_denominator = self.coeff_denominator.to(device)

        orig_dtype = x.dtype
        # If input is half precision, cast to float32 for safer kernel ops
        if orig_dtype == torch.float16:
            x = x.to(torch.float32)
            output = RationalsCUDAFunction.apply(x, coeff_numerator, coeff_denominator)
            output = output.to(orig_dtype)
        else:
            output = RationalsCUDAFunction.apply(x, coeff_numerator, coeff_denominator)
        return output


###############################################################################
#                    Init Function for the Activation                         #
###############################################################################
def init_rationals_cuda_activation(
    numerator_size: int = 5,
    denominator_size: int = 4,
    init: str = "normal",
    init_std: float = 1.0
) -> RationalsCUDAActivation:
    """
    Helper function to instantiate a RationalsCUDAActivation module
    with specified initialization parameters.

    Args:
        numerator_size (int): Number of coefficients in the numerator polynomial.
        denominator_size (int): Number of coefficients in the denominator polynomial.
        init (str): Initialization type ("normal" or "uniform").
        init_std (float): Std deviation for 'normal' init.

    Returns:
        RationalsCUDAActivation: A rational CUDA activation module initialized as requested.
    """
    return RationalsCUDAActivation(
        numerator_size=numerator_size,
        denominator_size=denominator_size,
        init=init,
        init_std=init_std
    )