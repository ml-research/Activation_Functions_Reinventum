#include <torch/extension.h>

#include <torch/extension.h>

// Declarations of CUDA functions
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

// C++ interface (forward and backward functions)
at::Tensor rationals_cuda_forward(
    at::Tensor x,
    at::Tensor coeff_numerator,
    at::Tensor coeff_denominator) {

    auto output = at::zeros_like(x);
    rationals_cuda_inline_forward(x, coeff_numerator, coeff_denominator, output);
    return output;
}

std::vector<at::Tensor> rationals_cuda_backward(
    at::Tensor grad_output,
    at::Tensor x,
    at::Tensor coeff_numerator,
    at::Tensor coeff_denominator) {

    auto grad_x = at::zeros_like(x);
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("rationals_cuda_forward", torch::wrap_pybind_function(rationals_cuda_forward), "rationals_cuda_forward");
m.def("rationals_cuda_backward", torch::wrap_pybind_function(rationals_cuda_backward), "rationals_cuda_backward");
}