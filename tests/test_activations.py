import pytest
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from activations.torch.classic_activations import ReLU, GLU, OneSin, Sigmoid, Tanh, LReLU
from activations.torch.learnable_activations.rationals import Rational, RARE
from activations.torch.learnable_activations.rationals.cuda_impl import RationalCUDA

from tests.utils import train_and_evaluate


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
def test_activation_training(activation_class, name):
    if activation_class is RationalCUDA:
        activation = activation_class()  # No name argument
    else:
        activation = activation_class(name=name)

    test_accuracy = train_and_evaluate(activation, name, num_epochs=1)
    assert test_accuracy > 5.0  # sanity check threshold
