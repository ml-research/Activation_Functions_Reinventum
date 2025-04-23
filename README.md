# 🧠 Activation Functions Benchmark

This repository provides a collection of classical and learnable activation functions implemented in PyTorch, including:

- 🧩 Standard: `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, etc.
- 🧪 Experimental: `GLU`, `OneSin`
- 📈 Learnable: `Rational`, `RARE`, and `RationalCUDA` (custom CUDA implementation)

Each activation can be dropped into PyTorch models and evaluated across various benchmarks, including training performance and gradient equivalence.

---

## 🚀 Performance Benchmarks on CIFAR-10

Model: Simple ConvNet with 2 Conv layers + 2 FC layers. All models were trained for **3 epochs**.

> **⚠️ Note**  
> The Rational-family activations (`Rational`, `RARE`, `RationalCUDA`) use randomly initialized parameters,  
> so their exact accuracies will vary between runs. These numbers are meant to be illustrative, not definitive.


| Activation    | Train Accuracy (%) | Test Accuracy (%) | Duration (s) |
|---------------|--------------------|--------------------|--------------|
| ReLU          | 71.70              | 67.99              | 13.68        |
| GLU           | 57.59              | 54.22              | 13.26        |
| OneSin        | 38.19              | 36.89              | 13.52        |
| Sigmoid       | 50.69              | 50.15              | 13.37        |
| Tanh          | 74.35              | 67.11              | 13.29        |
| LReLU         | 72.53              | 68.73              | 13.44        |
| Rational      | 68.95              | 63.31              | 21.79        |
| RARE          | 10.00              | 10.00              | 22.39        |
| RationalCUDA  | 64.82              | 60.59              | 13.57        |

---

## 🔬 Gradient Equivalence Test: Rational vs RationalCUDA

This test ensures that `RationalCUDA`, our custom CUDA-accelerated implementation, produces gradients similar to the CPU-based `Rational` function.

```markdown
| Metric             | Value          |
|--------------------|----------------|
| **MSE**            | `0.00030033`   |
| **Max Abs Diff**   | `0.0413`       |
| **Mean Abs Diff**  | `0.0106`       |
| **Test Result**    | ✅ **Passed** (tolerance = `3e-3`) |
```

> Minor numerical differences are expected due to floating-point precision but remain within acceptable tolerances.

---

## 📦 Module Structure

- `activations/torch/classic_activations/`: Classic non-learnable activations
- `activations/torch/learnable_activations/rationals/`: Learnable rational functions and CUDA backend
- `tests/`: Contains automated unit tests (training accuracy, performance, gradient tests)
- `utils/`: Shared model definitions and training helpers

---

## 🛠️ Installation

### Devcontainer (CUDA-enabled)

If you need the CUDA‐accelerated `RationalCUDA`, start the VS Code devcontainer and install inside it:

```bash
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . bash
```

Install requirements and the module inside the container
```bash
pip install -r requirements.txt
pip install -e .
```

ELSE: Local Virtual Environment (CPU or pre-configured CUDA)

```bash
python3 -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```
---

## 🔍 Running Tests

Run the full suite:

```bash
pytest -v tests/
```

```bash
pytest tests/test_performance.py
pytest tests/test_gradients_rationals.py::test_gradient_equivalence
```

---

## 👥 Authors

- Quentin Delfosse
- Patrick Schramowski
- Matthias Tichy

---

## 📜 License
MIT License

---
