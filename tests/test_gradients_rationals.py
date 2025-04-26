import torch
import pytest
import sys
import os
import random

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from activations.torch.learnable_activations.rationals import Rational
from activations.torch.learnable_activations.rationals.cuda_impl import RationalCUDA

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def compare_gradients(act_a, act_b, input_tensor, target_tensor, tol=None):
    device = input_tensor.device
    act_a = act_a.to(device)
    act_b = act_b.to(device)

    input_a = input_tensor.clone().detach().requires_grad_(True)
    input_b = input_tensor.clone().detach().requires_grad_(True)

    out_a = act_a(input_a)
    out_b = act_b(input_b)

    loss_fn = torch.nn.MSELoss()
    loss_a = loss_fn(out_a, target_tensor)
    loss_b = loss_fn(out_b, target_tensor)

    loss_a.backward()
    loss_b.backward()

    grad_diff = input_a.grad - input_b.grad
    mse = torch.mean(grad_diff ** 2).item()

    # Debug output
    print("\nGradient A:", input_a.grad)
    print("Gradient B:", input_b.grad)
    print(f"Gradient MSE: {mse:.8f}")
    print("Max abs diff:", grad_diff.abs().max().item())
    print("Mean abs diff:", grad_diff.abs().mean().item())

    return mse < tol


@pytest.mark.parametrize("act_class_a, act_class_b", [
    (Rational, RationalCUDA),
])
def test_gradient_equivalence(act_class_a, act_class_b):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)

    # Hardcoded coefficients
    numerator = torch.tensor([1.0, -0.5, 0.25, 0.1, -0.05], dtype=torch.float32)
    denominator = torch.tensor([1.0, 0.2, -0.1, 0.05], dtype=torch.float32)

    # Create activations
    act_a = act_class_a(name="test", degrees=(5, 4))
    act_b = act_class_b(numerator_size=5, denominator_size=4)

    # Override parameters manually to match
    with torch.no_grad():
        if hasattr(act_a, "numerator"):
            act_a.numerator.copy_(numerator.clone())
            act_a.denominator.copy_(denominator.clone())
        if hasattr(act_b, "coeff_numerator"):
            act_b.coeff_numerator.copy_(numerator.clone().to(torch.device("cuda")))
            act_b.coeff_denominator.copy_(denominator.clone().to(torch.device("cuda")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = (torch.rand(32, device=device) * 1.9 - 0.95)
    target_tensor = (torch.rand(32, device=device) * 1.9 - 0.95)

    assert compare_gradients(act_a, act_b, input_tensor, target_tensor, tol=3e-3), \
        f"Gradient mismatch between {act_class_a.__name__} and {act_class_b.__name__}"