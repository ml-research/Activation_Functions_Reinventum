"""
Utility script to build CUDA extensions for distribution.
This creates precompiled binaries for supported platforms.
"""
import os
import sys
import subprocess
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

# Ensure CUDA is available
if not torch.cuda.is_available() or CUDA_HOME is None:
    print("CUDA is not available. Cannot build CUDA extensions.")
    sys.exit(1)

# Set architectures for compatibility
os.environ["TORCH_CUDA_ARCH_LIST"] = "3.5;5.0;6.0;7.0;7.5;8.0+PTX"

# Get CUDA version
cuda_version = torch.version.cuda
print(f"Building CUDA extensions for CUDA version {cuda_version}")

# Path to the extension source files
ext_dir = os.path.join("activations", "torch", "learnable_activations", "rationals", "cuda_impl")
os.makedirs(ext_dir, exist_ok=True)

# Ensure the source files exist
sources = [
    os.path.join(ext_dir, "cuda_impl.cpp"),
    os.path.join(ext_dir, "cuda_impl_kernel.cu"),
]

for src in sources:
    if not os.path.exists(src):
        print(f"ERROR: Source file missing: {src}")
        sys.exit(1)

# Remove old build files
subprocess.run(["rm", "-rf", "build", "dist", "activations_cuda_build.egg-info"], check=False)

# Define paths
ext_dir = os.path.join("activations", "torch", "learnable_activations", "rationals", "cuda_impl")
sources = [
    os.path.join(ext_dir, "cuda_impl.cpp"),
    os.path.join(ext_dir, "cuda_impl_kernel.cu"),
]

# Ensure source files exist
for src in sources:
    if not os.path.exists(src):
        print(f"ERROR: Missing source file: {src}")
        sys.exit(1)

# CUDA Compilation Flags
extra_compile_args = {
    "cxx": ["-O3"],  # Verbose output for C++
    "nvcc": [
        "-O3",
        "--use_fast_math",
        "-Xcompiler", "-fPIC",
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90,code=sm_90",
        "-gencode=arch=compute_90,code=compute_90",
        "-v"  # Enable verbose output
    ],
}

# Build Extension
ext_modules = [
    CUDAExtension(
        name="activations.torch.learnable_activations.rationals.cuda_impl.rational_cuda",
        sources=sources,
        extra_compile_args=extra_compile_args,
    )
]

# Setup
setup(
    name="activations_cuda_build",
    version="0.1",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)

print("✅ CUDA extension successfully built.")