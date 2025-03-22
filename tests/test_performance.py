import pytest
import sys
import os
import csv

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from activations.torch import ReLU, GLU, OneSin, Sigmoid, Tanh, LReLU
from activations.torch.learnable_activations.rationals import Rational, RARE
from activations.torch.learnable_activations.rationals.cuda_impl import RationalCUDA
from tests.utils import train_and_evaluate_perf

LOG_FILE = "performance.txt"

# Create the file with headers if it doesn't exist
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Activation", "Train Accuracy (%)", "Test Accuracy (%)", "Duration (s)"])

@pytest.mark.parametrize("activation_class, name", [
    (ReLU, "ReLU"),
    (GLU, "GLU"),
    (OneSin, "OneSin"),
    (Sigmoid, "Sigmoid"),
    (Tanh, "Tanh"),
    (LReLU, "LReLU"),
    (Rational, "Rational"),
    (RARE, "RARE"),
    (RationalCUDA, "RationalCUDA"),
])
def test_activation_performance(activation_class, name):
    activation = activation_class() if name == "RationalCUDA" else activation_class(name=name)
    duration, train_acc, test_acc = train_and_evaluate_perf(activation, name, num_epochs=3)
    
    print(f"[{name}] Time: {duration:.2f}s, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

    # Append results to log file
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([name, f"{train_acc:.2f}", f"{test_acc:.2f}", f"{duration:.2f}"])

    # Pass/fail test based on minimum test accuracy
    assert test_acc > 10, f"{name} failed to reach basic accuracy threshold"
