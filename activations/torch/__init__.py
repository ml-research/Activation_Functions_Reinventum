import os

from .activation_module import ActivationModule
from .learnable_activations import BiTentActivation, TentActivation, Rational, RARE, EmbeddedRational, rationals
from .classic_activations import ReLU, LReLU, Tanh, Sigmoid, GLU, OneSin

ActivationModule.set_plotting_style(path=os.path.join(os.path.dirname(__file__), "modified_whitegrid.mplstyle"))