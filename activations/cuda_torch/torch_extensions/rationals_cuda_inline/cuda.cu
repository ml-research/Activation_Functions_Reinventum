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

#define M 5
#define N 4

// Forward declarations and helper functions
template <typename scalar_t>
__device__ __forceinline__ scalar_t fused_mul_add(scalar_t a, scalar_t b, scalar_t c) {
    return a * b + c;
}

template <>
__device__ __forceinline__ at::Half fused_mul_add(at::Half a, at::Half b, at::Half c) {
    return at::Half(float(a) * float(b) + float(c));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t abs_val(scalar_t x) {
    return fabs(x);
}

template <>
__device__ __forceinline__ at::Half abs_val(at::Half x) {
    return at::Half(fabs(float(x)));
}

// AtomicAdd wrapper for different types
template <typename T>
__device__ __forceinline__ void atomicAddWrapper(T* address, T val);

// Specialize for float
template <>
__device__ __forceinline__ void atomicAddWrapper<float>(float* address, float val) {
    atomicAdd(address, val);
}

// Specialize for double
template <>
__device__ __forceinline__ void atomicAddWrapper<double>(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    atomicAdd(address, val);
#else
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                       __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
#endif
}

// Specialize for half using a union for type-punning
template <>
__device__ __forceinline__ void atomicAddWrapper<at::Half>(at::Half* address, at::Half val) {
    // Fix pointer arithmetic by casting to char* first
    unsigned int* address_as_ui = (unsigned int*)(((char*)address) - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    do {
        assumed = old;
        
        // Use a union to safely convert at::Half to __half_raw
        union {
            at::Half a;
            __half h;
        } u;
        u.a = val;
        __half_raw raw_val = __half_raw(u.h);
        
        unsigned int new_val;
        if (((size_t)address & 2)) {
            new_val = (old & 0x0000FFFF) | (raw_val.x << 16);
        } else {
            new_val = (old & 0xFFFF0000) | raw_val.x;
        }
        old = atomicCAS(address_as_ui, assumed, new_val);
    } while (assumed != old);
}

// Forward kernel
template <typename scalar_t>
__global__ void rationals_cuda_inline_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ coeff_num,
    const scalar_t* __restrict__ coeff_den,
    scalar_t* __restrict__ output,
    int num_elements) {

    // Shared memory for coefficients
    __shared__ scalar_t shared_coeff_num[M];
    __shared__ scalar_t shared_coeff_den[N];

    // Load coefficients into shared memory (reverse order)
    if (threadIdx.x < M) {
        shared_coeff_num[threadIdx.x] = coeff_num[M - 1 - threadIdx.x];
    }
    if (threadIdx.x < N) {
        shared_coeff_den[threadIdx.x] = coeff_den[N - 1 - threadIdx.x];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

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

        denominator = denominator * x_val;
        scalar_t denom = abs_val(denominator) + scalar_t(1.0);
        output[idx] = numerator / denom;
    }
}

// Backward kernel
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

    using accscalar_t = at::acc_type<scalar_t, true>;

    __shared__ scalar_t shared_coeff_num[M];
    __shared__ scalar_t shared_coeff_den[N];

    if (threadIdx.x < M) {
        shared_coeff_num[threadIdx.x] = coeff_num[M - 1 - threadIdx.x];
    }
    if (threadIdx.x < N) {
        shared_coeff_den[threadIdx.x] = coeff_den[N - 1 - threadIdx.x];
    }
    __syncthreads();

    accscalar_t local_grad_coeff_num[M] = {0};
    accscalar_t local_grad_coeff_den[N] = {0};

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; idx < num_elements; idx += stride) {
        scalar_t x_val = x[idx];
        scalar_t grad_out = grad_output[idx];

        scalar_t numerator = shared_coeff_num[0];
        scalar_t numerator_derivative = scalar_t(0);
        #pragma unroll
        for (int i = 1; i < M; ++i) {
            numerator_derivative = fused_mul_add(numerator_derivative, x_val, numerator);
            numerator = fused_mul_add(numerator, x_val, shared_coeff_num[i]);
        }

        scalar_t Q = shared_coeff_den[0];
        scalar_t Q_prime = scalar_t(0);
        #pragma unroll
        for (int i = 1; i < N; ++i) {
            Q_prime = fused_mul_add(Q_prime, x_val, Q);
            Q = fused_mul_add(Q, x_val, shared_coeff_den[i]);
        }

        scalar_t Q_tilde = Q * x_val;
        scalar_t Q_tilde_prime = Q_prime * x_val + Q;
        scalar_t denom = abs_val(Q_tilde) + scalar_t(1.0);
        scalar_t sign_Q_tilde = (Q_tilde >= scalar_t(0)) ? scalar_t(1.0) : scalar_t(-1.0);
        scalar_t denom_derivative = sign_Q_tilde * Q_tilde_prime;

        scalar_t denom_squared = denom * denom;
        scalar_t grad_input = (numerator_derivative * denom - numerator * denom_derivative) / denom_squared;
        grad_x[idx] = grad_out * grad_input;

        scalar_t inv_denom = scalar_t(1.0) / denom;
        scalar_t inv_denom_squared = inv_denom * inv_denom;
        scalar_t common_factor_num = grad_out * inv_denom;
        scalar_t common_factor_den = -grad_out * numerator * sign_Q_tilde * inv_denom_squared;

        scalar_t accum = scalar_t(1.0);
        #pragma unroll
        for (int i = 0; i < M; ++i) {
            local_grad_coeff_num[i] += accscalar_t(accum * common_factor_num);
            accum *= x_val;
        }

        accum = x_val;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            local_grad_coeff_den[i] += accscalar_t(accum * common_factor_den);
            accum *= x_val;
        }
    }

    extern __shared__ char shared_mem_raw[];
    accscalar_t* shared_grad = reinterpret_cast<accscalar_t*>(shared_mem_raw);
    accscalar_t* shared_grad_num = shared_grad;
    accscalar_t* shared_grad_den = shared_grad + M * blockDim.x;

    for (int i = 0; i < M; ++i) {
        shared_grad_num[i * blockDim.x + threadIdx.x] = local_grad_coeff_num[i];
    }
    for (int i = 0; i < N; ++i) {
        shared_grad_den[i * blockDim.x + threadIdx.x] = local_grad_coeff_den[i];
    }
    __syncthreads();

    for (int i = 0; i < M; ++i) {
        accscalar_t* sdata = &shared_grad_num[i * blockDim.x];
        unsigned int tid = threadIdx.x;
        
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        
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

void rationals_cuda_inline_forward(
    at::Tensor x,
    at::Tensor coeff_numerator,
    at::Tensor coeff_denominator,
    at::Tensor output) {

    // Ensure tensors are contiguous
    x = x.contiguous();
    coeff_numerator = coeff_numerator.contiguous();
    coeff_denominator = coeff_denominator.contiguous();
    output = output.contiguous();

    auto x_flat = x.view(-1);
    auto output_flat = output.view(-1);
    int num_elements = x_flat.size(0);

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rationals_cuda_inline_forward", ([&] {
        rationals_cuda_inline_forward_kernel<scalar_t><<<blocks, threads>>>(
            x_flat.data_ptr<scalar_t>(),
            coeff_numerator.data_ptr<scalar_t>(),
            coeff_denominator.data_ptr<scalar_t>(),
            output_flat.data_ptr<scalar_t>(),
            num_elements);
    }));
}

void rationals_cuda_inline_backward(
    at::Tensor grad_output,
    at::Tensor x,
    at::Tensor coeff_numerator,
    at::Tensor coeff_denominator,
    at::Tensor grad_x,
    at::Tensor grad_coeff_numerator,
    at::Tensor grad_coeff_denominator) {

    grad_output = grad_output.contiguous();
    x = x.contiguous();
    coeff_numerator = coeff_numerator.contiguous();
    coeff_denominator = coeff_denominator.contiguous();
    grad_x = grad_x.contiguous();
    grad_coeff_numerator = grad_coeff_numerator.contiguous();
    grad_coeff_denominator = grad_coeff_denominator.contiguous();

    auto x_flat = x.view(-1);
    auto grad_output_flat = grad_output.view(-1);
    auto grad_x_flat = grad_x.view(-1);
    int num_elements = x_flat.size(0);

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

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
