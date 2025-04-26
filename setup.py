from pathlib import Path
from setuptools import setup, find_packages
from distutils.command.clean import clean
import os
import sys
import subprocess
import platform
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

degrees = [(5, 4), (7, 6)]
name = 'activations'
long_description = ""

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.readlines()

# Get the path to the current file
current_path = Path(__file__).parent.absolute()

# Define the path to the compiled module
cuda_extension_path = current_path / "activations" / "torch" / "learnable_activations" / "rationals" / "cuda_impl"

# Define the sources for the CUDA extension
cuda_sources = [
    str(cuda_extension_path / "cuda_impl.cpp"),
    str(cuda_extension_path / "cuda_impl_kernel.cu"),
]

# Check if CUDA is available
has_cuda = torch.cuda.is_available() and CUDA_HOME is not None

# Function to check if we need to build the CUDA extension
def need_cuda_build():
    if not has_cuda:
        return False
    
    # Check if the compiled module already exists
    # This path should match the expected output from BuildExtension
    lib_path = list(current_path.glob("build/lib*/activations/torch/learnable_activations/rationals/cuda_impl/*.so"))
    lib_path.extend(list(current_path.glob("build/lib*/activations/torch/learnable_activations/rationals/cuda_impl/*.pyd")))
    
    if lib_path:
        return False
    
    return True

# Setup the CUDA extension if needed
ext_modules = []
if need_cuda_build():
    ext_modules.append(
        CUDAExtension(
            name='activations.torch.learnable_activations.rationals.cuda_impl.rational_cuda',
            sources=cuda_sources,
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': ['-O2', '--use_fast_math']
            }
        )
    )

# Clean command to remove build artifacts
class clean_all(clean):
    def run(self):
        self.all = True
        super().run()
        import shutil
        egginf = name.replace('-', '_')
        
        # Remove these directories if they exist
        for dir_to_clean in [f"{egginf}.egg-info", "dist", "build"]:
            if os.path.exists(dir_to_clean):
                shutil.rmtree(dir_to_clean)
        
        print("Cleaned everything")

setup(
    name=name,
    version="1.0.0",
    author="Quentin Delfosse, Patrick Schramowski, & Matthias Tichy",
    author_email="quentin.delfosse@cs.tu-darmstadt.de",
    description="Activations functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/k4ntz/activation_functions",
    packages=find_packages(exclude=["tests"]),
    package_data={
        '': ['*.json', '*.so', '*.pyd'],  # Include precompiled binaries
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License"
    ],
    install_requires=requirements,
    ext_modules=ext_modules,
    cmdclass={
        'clean': clean_all,
        'build_ext': BuildExtension
    },
    python_requires='>=3.5.0',
)