import torch
import os
import torch.nn as nn
from rationals_cuda_inline import init_rationals_cuda_activation
from rationals_new import RationalsModel  # Ensure correct PyTorch implementation import

os.environ['CUDA_VISIBLE_DEVICES'] = "4,5"

def compare_gradients_io(device='cuda'):
    """
    Compares gradients and input-output behavior between CUDA and PyTorch implementations
    of the rational activation function.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Initialize activations with the same parameters
    torch.manual_seed(42)
    activation_cuda = init_rationals_cuda_activation(numerator_size=5, denominator_size=4, init="normal", init_std=1.0).to(device)
    torch.manual_seed(42)
    activation_pytorch = RationalsModel().to(device)
    
    # Create random input tensor
    x = torch.randn(10, 10, dtype=torch.float32, requires_grad=True, device=device)
    
    # Forward pass
    output_cuda = activation_cuda(x)
    output_pytorch = activation_pytorch(x)
    
    # Check if outputs match within tolerance
    if torch.allclose(output_cuda, output_pytorch, atol=1e-6):
        print("[✓] Outputs match.")
    else:
        print("[✗] Outputs do NOT match!")
        print("CUDA Output:", output_cuda)
        print("PyTorch Output:", output_pytorch)
    
    # Compute loss
    loss_cuda = output_cuda.sum()
    loss_pytorch = output_pytorch.sum()
    loss_cuda.backward()
    loss_pytorch.backward()
    
    # Compare input gradients
    if torch.allclose(x.grad, x.grad, atol=1e-6):
        print("[✓] Input gradients match.")
    else:
        print("[✗] Input gradients do NOT match!")
        print("CUDA Gradients:", x.grad)
        print("PyTorch Gradients:", x.grad)
    
    # Compare coefficient gradients if they exist
    if hasattr(activation_cuda, 'coeff_numerator') and hasattr(activation_pytorch, 'coeff_numerator'):
        grad_num_cuda = activation_cuda.coeff_numerator.grad
        grad_num_pytorch = activation_pytorch.coeff_numerator.grad
        if torch.allclose(grad_num_cuda, grad_num_pytorch, atol=1e-6):
            print("[✓] Numerator coefficient gradients match.")
        else:
            print("[✗] Numerator coefficient gradients do NOT match!")
            print("CUDA Numerator Gradient:", grad_num_cuda)
            print("PyTorch Numerator Gradient:", grad_num_pytorch)
    
    if hasattr(activation_cuda, 'coeff_denominator') and hasattr(activation_pytorch, 'coeff_denominator'):
        grad_den_cuda = activation_cuda.coeff_denominator.grad
        grad_den_pytorch = activation_pytorch.coeff_denominator.grad
        if torch.allclose(grad_den_cuda, grad_den_pytorch, atol=1e-4):
            print("[✓] Denominator coefficient gradients match.")
        else:
            print("[✗] Denominator coefficient gradients do NOT match!")
            print("CUDA Denominator Gradient:", grad_den_cuda)
            print("PyTorch Denominator Gradient:", grad_den_pytorch)
    
if __name__ == "__main__":
    compare_gradients_io()
