#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
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