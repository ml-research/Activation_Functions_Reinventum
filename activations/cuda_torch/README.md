# Rational Activation Function CUDA

A **custom CUDA-accelerated** activation function for PyTorch, implemented as a rational function (polynomial ratio) using **Horner's method** for efficient forward and backward passes. Additionally, this repository contains **modular** scripts for:

1. **Data loading** for Imagenette (or other datasets).  
2. **Model definition** (a simple Vision Transformer) that can **use** the rational activation.  
3. **Profiling and benchmarking** scripts to measure performance with different activations.  
4. **Visualization** scripts to analyze and compare results.

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Repository Structure](#repository-structure)  
3. [Features](#features)  
4. [Installation](#installation)  
   - [Prerequisites](#prerequisites)  
   - [Steps](#steps)  
5. [Usage](#usage)  
   - [Data Loading](#data-loading)  
   - [Model Creation](#model-creation)  
   - [Profiling & Benchmarking](#profiling--benchmarking)  
   - [Visualization](#visualization)  
6. [Development Environment](#development-environment)  
   - [Using Visual Studio Code Remote Containers](#using-visual-studio-code-remote-containers)  
   - [Building the Docker Image Manually](#building-the-docker-image-manually)  
7. [Files](#files)  
8. [License](#license)

---

## Introduction

This project provides a **CUDA kernel** implementation of a **rational activation** function for use in PyTorch neural networks. The activation function is generally defined as:

\[
f(x) = \frac{\sum_{i=0}^{M-1} a_i\, x^i}{ 1 + \left|\sum_{j=1}^{N-1} b_j\, x^j\right| }
\]

where \(a_i\) and \(b_j\) are learnable coefficients, and \(M\) and \(N\) are the degrees of the numerator and denominator polynomials, respectively. The code uses **Horner's method** for efficient polynomial evaluation and **atomic operations** in the backward pass for safe gradient accumulation across threads.

---

## Repository Structure

├── devcontainer.json # VS Code Remote Container configuration 

├── data_loader.py # Prepares Imagenette (or custom) datasets and DataLoaders

├── model_loader.py # Defines the Vision Transformer (SimpleViT) and a helper to create it 

├── rationals_cuda_inline.py # Core CUDA-based activation kernel logic, load_inline compilation, init functions 

├── test_cuda_activation_profiler.py (or a combined script) │ # Script that trains & profiles with the Rational CUDA activation

├── visualize.py # Gathers profiler logs, JSON results, and produces plots 

├── requirements.txt # Python dependencies 

├── README.md # (This file) 




---

## Features

- **CUDA Acceleration**: Leverages GPU parallelism for efficient forward/backward passes.  
- **Customizable**: Easily change numerator or denominator degrees (\(M, N\)) and initialization.  
- **PyTorch Integration**: Can be used as a drop-in `nn.Module` activation (like ReLU, etc.).  
- **Optimized Computation**: Horner's method, fused multiply-add, atomicAdd wrappers for half, float, and double.

---

## Installation

### Prerequisites

- **NVIDIA GPU**: A CUDA-capable GPU with recent drivers installed.  
- **CUDA Toolkit**: Ensure it matches your environment and PyTorch version.  
- **PyTorch**: Installed with CUDA support (`torch>=2.0.0`).  
- (Optional) **Docker** and **Visual Studio Code** (with Remote - Containers extension) for a consistent dev setup.

### Steps

1. **Check CUDA and Driver Versions**

   ```bash
   nvidia-smi
   ```

Confirm that you see a driver and CUDA version compatible with your system.

2. **(Optional) Install CUDA Toolkit inside a Docker container**  
If you use the provided Dockerfile or devcontainer, it will handle installing a suitable CUDA Toolkit. Otherwise, follow the appropriate instructions to install the CUDA toolkit on your host or container.

3. **Install Python Dependencies**  
If you have a requirements.txt:

    ```bash
    pip install -r requirements.txt
    ```
4. **Compile the Extension**  
The first time you import and use `rationals_cuda_inline.py`, PyTorch will call `load_i

## Usage

**Data Loading**

`data_loader.py` (Example)  
A script/module that provides a function like `get_imagenette_dataloaders(...)`, returning PyTorch `DataLoader`s for training/validation/testing.

**Example**:
```python
from data_loader import get_imagenette_dataloaders

train_loader, val_loader, test_loader = get_imagenette_dataloaders(
    image_path="/path/to/imagenette",
    img_size=160,
    batch_size=16
)
```

**Model Creation:**
model_loader.py defines a SimpleViT (Vision Transformer) that can optionally use a separate optimizer for the rational coefficients vs. the main model.
You can create the model via create_simple_vit(...) or define your own.

**Example**:
```python
from model_loader import create_simple_vit

model = create_simple_vit(
    image_size=160,
    patch_size=16,
    num_classes=10,
    lr=5e-5,
    # ...
)
```

**Profiling & Benchmarking**
test_cuda_activation_profiler.py (or an alternative script like compare_activation_performance.py) runs training under PyTorch’s Profiler and logs performance data:

- Trains the model on the specified data loaders.
- Logs GPU usage, timings, memory usage, etc.
- Optionally compares the CUDA-based rational activation to others (e.g., ReLU, GELU) if you want to see performance differences.

**Example usage**:
```bash
python3 test_cuda_activation_profiler.py
```
This might produce logs in ./log/profiler/ and store final metrics in a JSON file (e.g., unified_activation_results.json).

**Visualization**
visualize_activations.py (or similar) can parse:

The .txt profiler reports in ./log/profiler/.
A JSON results file containing metrics like validation accuracy, training time, average inference time, etc.
TensorBoard event logs (if you used a TensorBoardLogger in PyTorch Lightning).
It then aggregates or plots bar charts, line charts, etc. in ./visualizations/ to help you compare different runs/configurations.

**Example usage**:
```bash
python visualize_activations.py
```

## Development Environment

**Using Visual Studio Code Remote Containers**
1. Install the **Remote - Containers** extension in VS Code.  
2. Open the project folder in VS Code.  
3. When prompted, reopen the folder in a container.  
4. The container will build automatically using the provided Dockerfile (`Dockerfile`) and `devcontainer.json`.

---

## Building the Docker Image Manually

1. **Build** the Docker image:
   ```bash
   docker build -t rational-cuda-activation .

2. **Run** the container with GPU access:
```bash
docker run --gpus all -it rational-cuda-activation
```

3. Inside the container, run any of the scripts (e.g. test_cuda_activation_profiler.py) to compile the extension, train a model, etc.

## Files

- **`rationals_cuda_inline.py`**  
  Core logic for compiling and loading the CUDA extension via `torch.utils.cpp_extension.load_inline`.  
  Contains `RationalsCUDAFunction` and `RationalsCUDAActivation` classes, plus a helper like `init_rationals_cuda_activation(...)`.

- **`data_loader.py`**  
  Functions to prepare `DataLoader`s (e.g., for Imagenette).

- **`model_loader.py`**  
  Defines the `SimpleViT` class (LightningModule) and a helper `create_simple_vit(...)` to build it.

- **`test_cuda_activation_profiler.py`** or **`compare_activation_performance.py`**  
  Scripts that train and profile the model, possibly comparing multiple activations, logging performance stats.

- **`visualize_activations.py`**  
  Aggregates profiler logs (`.txt`) and JSON results (training time, accuracy, etc.).  
  Produces bar plots, line charts, etc., saved to `./visualizations/`.

- **`requirements.txt`**  
  Python dependencies (e.g., `torch>=2.0.0`, `torchvision`, etc.).

- **`Dockerfile`, `devcontainer.json`**  
  Container environment setup. Installs CUDA, PyTorch, and other dependencies.

- **`README.md`**  
  This file, explaining the project and usage.



## License

This project is licensed under the MIT License. See the LICENSE file for details (or adapt the license text as needed).