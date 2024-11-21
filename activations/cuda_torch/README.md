# Rational Activation Function CUDA

A custom CUDA-accelerated activation function for PyTorch, implemented as a rational function using Horner's method for efficient computation.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Steps](#steps)
    - [1. Check CUDA and Driver Versions](#1-check-cuda-and-driver-versions)
    - [2. Install CUDA Toolkit 12.1 Manually Inside the Container (Example)](#2-install-cuda-toolkit-121-manually-inside-the-container-example)
- [Usage](#usage)
- [Testing](#testing)
- [Development Environment](#development-environment)
  - [Using Visual Studio Code Remote Containers](#using-visual-studio-code-remote-containers)
  - [Building the Docker Image Manually](#building-the-docker-image-manually)
- [Files](#files)
  - [devcontainer.json](#devcontainerjson)
  - [Dockerfile](#dockerfile)
  - [requirements.txt](#requirementstxt)
- [License](#license)

## Introduction

This project provides a CUDA kernel implementation of a rational activation function for use in PyTorch neural networks. The activation function is defined as:

$$
f(x) = \frac{\sum_{i=0}^{M-1} a_i x^i}{1 + \left| \sum_{j=1}^{N-1} b_j x^j \right|}
$$

where $a_i$ and $b_j$ are learnable coefficients, and $M$ and $N$ are the degrees of the numerator and denominator polynomials, respectively.

## Features

- **CUDA Acceleration**: Leverages GPU computation for efficient forward and backward passes.
- **Customizable**: Parameters \( M \) and \( N \) define the degrees of the numerator and denominator polynomials.
- **PyTorch Integration**: Easily integrates with PyTorch models as a drop-in activation function.
- **Optimized Computation**: Uses Horner's method and fused multiply-add operations for numerical stability and performance.

## Installation

### Prerequisites

- **NVIDIA GPU**: Compatible GPU with necessary drivers installed.
- **CUDA Toolkit**: Version compatible with your GPU and PyTorch installation.
- **PyTorch**: Installed with CUDA support.
- **Docker**: If using the Docker environment.
- **Visual Studio Code**: With the Remote - Containers extension (optional).

### Steps

1. Check CUDA and Driver Versions

    Before proceeding, verify your CUDA and driver versions using `nvidia-smi`:

    ```bash
    nvidia-smi
    ```

    ### Example Output:
    ```
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 535.183.06   Driver Version: 535.183.06   CUDA Version: 12.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    | ...                                                                     ... |
    +-----------------------------------------------------------------------------+
    ```
2. Install CUDA Toolkit 12.1 Manually Inside the Container (Example)

    *Note: The following steps are specific to CUDA Toolkit 12.1 and Ubuntu 22.04. Adjust the commands based on your CUDA version and operating system.*

    **a. Download the CUDA Toolkit 12.1 Installer:**

    ```bash
    wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb
    ```
    **b. Install the CUDA Repository Package:**

    ```bash
    dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb
    ```
    **c. Add the GPG Key:**

    ```bash
    cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
    ```
    **d. Update Package Lists:**

    ```bash
    apt-get update
    ```
    **e. Install the CUDA Toolkit:**

    ```bash
    apt-get install -y cuda-toolkit-12-1
    ```
    **f. Update Environment Variables:**

    ```bash
    echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
    ```
    **g. Verify Installation:**

    ```bash
    which nvcc
    nvcc --version
    ```
    *Expected Output:*

    *nvcc should be found in /usr/local/cuda-12.1/bin/nvcc. nvcc --version should display CUDA Toolkit 12.1.*

### Usage

```
import torch
import torch.nn as nn
from rationals_cuda_inline import RationalsCUDAActivation

# Define your model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 10)
        self.activation = RationalsCUDAActivation()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

# Instantiate and use the model
model = MyModel().cuda()
input_data = torch.randn(32, 10).cuda()
output = model(input_data)
```

### To run the test function provided:

```
python rationals_cuda_inline.py
```

*Ensure that your CUDA environment is properly configured and that you have a compatible GPU.*

### Development Environment

*For a consistent development environment, you can use the provided devcontainer.json and Dockerfile. This sets up a container with all necessary dependencies.*

## Using Visual Studio Code Remote Containers

1. Install the Remote - Containers extension in VS Code.
2. Open the project folder in VS Code.
3. When prompted, reopen the folder in a container.
4. The container will build automatically using the provided Dockerfile.

## Building the Docker Image Manually
1. Build the Docker image
```
docker build -t rational-cuda-activation .
```
2. Run the Docker container
```
docker run --gpus all -it rational-cuda-activation
```
*The --gpus all flag ensures that the container has access to your GPU.*

3. Inside the Container:
```
python rationals_cuda_inline.py
```

### Files
*devcontainer.json*
```
{
    "name": "Rational CUDA Activation",
    "image": "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime",
    "runArgs": [
        "--shm-size=16g",
        "--gpus=all",
        "--cpuset-cpus=0-19"
    ],
    "postCreateCommand": "pip install -r requirements.txt",
    "customizations": {
        "vscode": {
            "extensions": [
                "ms-python.python",
                "ms-vscode.cpptools",
                "ms-azuretools.vscode-docker"
            ]
        }
    }
}
 
```

## Dockerfile
```
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    libopenblas-dev \
    wget \
    ninja-build \  # Added ninja-build
    && rm -rf /var/lib/apt/lists/*

# Install CUDA Toolkit 12.4
RUN wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-530.61.05-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-530.61.05-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && apt-get install -y cuda-toolkit-12-4 && \
    rm cuda-repo-ubuntu2204-12-4-local_12.4.0-530.61.05-1_amd64.deb && \
    rm -rf /var/lib/apt/lists/*

# Create symbolic links for Python and pip
RUN ln -s /usr/bin/python3.10 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA 12.1 support (from the base image)
# Note: Ensure compatibility between CUDA 12.1 and 12.4 if needed
# If CUDA 12.4 is required for specific operations, verify PyTorch compatibility

# Copy the current directory into the container
COPY . /workspace

# Install Python dependencies
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda-12.4
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
ENV PATH=${CUDA_HOME}/bin:${PATH}

# Default command
CMD ["/bin/bash"]
```
*Note: The Dockerfile includes PyTorch with CUDA 12.1 support. Adjust the base image and installation steps if you need a different CUDA version.

## Build and Run Docker image:

```
# Build the Docker image
docker build -t rational-cuda-activation .

# Run the Docker container
docker run --gpus all -it rational-cuda-activation
```

## requirements.txt
```
torch>=2.0.0
torchvision>=0.15.0
```
*Adjust versions as necessary based on compatibility with your environment.*

### License

*This project is licensed under the MIT License.*