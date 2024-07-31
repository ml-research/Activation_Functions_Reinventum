"""
Rational Activation Functions for Pytorch
=========================================

This module allows you to create Rational Neural Networks using Learnable
Rational activation functions with Pytorch networks.
"""
import numpy as np
import torch
import torch.nn as nn
import scipy.optimize

from activations.torch.activation_module import ActivationModule
from activations.utils.utils import find_closest_equivalent
from activations.utils.rational_json import JsonHandler
from activations.torch.learnable_activations.rationals.functions import rational_A, rational_B, rational_C, rational_D, rational_nonsafe, rational_spline



def _get_rational_fn(version):
    if version == "A":
        return rational_A
    elif version == "B":
        return rational_B
    elif version == "C":
        return rational_C
    elif version == "D":
        return rational_D
    elif version == "N":
        return rational_nonsafe
    elif version == "S":
        return rational_spline
    else:
        raise ValueError(f"Unsupported version, got {version} expected one of [A, B, C, D, N, S]")


def find_weights(func, x, degrees, version, **scipyargs):
    """Approximates a function with a rational.
    
    Args:
        func (callable): The target function.
        x (array_like): Input data.
        degrees (tuple(int, int)): Number of weights for (numerator, denominator).
        scipyargs: Additional arguments passed to ``scipy.optimize.cuve_fit`` (method is 'lm').
            The following arguments can not be passed this way ``f, xdata, ydata, p0, method, full_output``.
    
    Returns:
        w_numerator, w_denominator: Found weights for numerator/denominator.
    """
    n_num, n_denom = degrees
    if version == "RARE":
        n_num -= 2
    elif version == "C":
        n_denom += 1
    n_total = n_num + n_denom + 1

    rat_fn = _get_rational_fn(version)
    if version == "RARE":
        k1 = x - torch.abs(x)
        k2 = x + torch.abs(x)
        def rational(x, params):
            y = rat_fn(x, params[:n_num], params[n_num:])
            return y * k1 * k2
    else:
        def rational(x, params):
            return rat_fn(x, params[:n_num], params[n_num:])

    if version == "RARE":
        w_init = torch.rand(n_total)
    else:
        w_init = torch.ones(n_total)

    params = scipy.optimize.curve_fit(
        f=rational,
        xdata=x,
        ydata=func(x),
        p0=w_init,
        method="lm",
        full_output=False,
        **scipyargs
    )[0]

    w_numerator, w_denominator = params[:n_num], params[n_num:]
    return w_numerator, w_denominator


class RationalBase(nn.Module):
    def __init__(self, name, group, logger=None):
        super().__init__()
        ActivationModule.register(self, name=name, group=group, logger=logger)
        
        self.best_fitted_function = None

    def fit(self, function, x=None):
        """
        Compute the parameters a, b, c, and d to have the neurally equivalent 
        function of the provided one as close as possible to this rational
        function.

        Arguments:
                function (callable):
                    The function you want to fit to rational.
                x (array):
                    The range on which the curves of the functions are fitted
                    together.
                    Default ``None``
        Returns:
            tuple: ((a, b, c, d), dist) with:
            a, b, c, d: the parameters to adjust the function
                (vertical and horizontal scales and bias)
            dist: The final distance between the rational function and the
            fitted one
        """
        if x is None:
            x = np.arange(-3., 3., 0.1)
        (a, b, c, d), distance = find_closest_equivalent(function, x)
        return (a, b, c, d), distance

    def best_fit(self, functions, x=None):
        """
        Compute the distance between the rational and the functions in
        :param:``functions``, and return the one with the minimal the distance.

        Arguments:
                functions_list (list of callable):
                    The function you want to fit to rational.
                x (array):
                    The range on which the curves of the functions are fitted
                    together.
                    Default ``None``
        Returns:
            tuple: ((a, b, c, d), dist) with:
            a, b, c, d: the parameters to adjust the function
                (vertical and horizontal scales and bias)
            dist: The final distance between the rational function and the
            fitted one
        """
        best_params, min_dist = self.fit(functions[0], x=x)
        best_func = functions[0]
        for func in functions[1:]:
            (a, b, c, d), distance = self.fit(func, x=x)
            if min_dist > distance:
                min_dist = distance
                best_params = (a, b, c, d)
                best_func = func
        return best_func, best_params


class Rational(RationalBase):
    """A Rational activation function.
    
    Args:
        init (str, callable or None): Method of weight initiaization.
            * If of type ``str``, must be a known activation function which will be approximated.
            * If ``callable`` must have two arguments. The first is the number of weights (for denominator or numerator)
                and the second is ``True`` if numerator weights are generated and ``False`` for denominator. Should return
                a ``torch.Tensor``.
            * If ``None`` numerator and denominator will be initialized uniformly in `[0, 1]`.
            Default ``None``.
    """
    def __init__(self, init=None, degrees=(5, 4), device="cpu",
                 version="A", train_numerator=True, train_denominator=True,
                 name="Rational", group=None, logger=None, **kwargs):
        super().__init__(name=name, group=group, logger=logger)

        n_num, n_denom = degrees
        if isinstance(init, str):
            w_numerator, w_denominator = JsonHandler.load(version, degrees, init)
        elif init is None:
            w_numerator = torch.rand(n_num)
            w_denominator = torch.rand(n_denom)
        else:
            w_numerator = init(n_num, True)
            w_denominator = init(n_denom, False)

        self.numerator = nn.Parameter(torch.tensor(w_numerator, device=device), requires_grad=train_numerator)
        self.denominator = nn.Parameter(torch.tensor(w_denominator, device=device), requires_grad=train_denominator)
        
        self.degrees = degrees
        self.version = version
        self.init_approximation = f"{init}"

        self.version_kwargs = {}
        self.activation_function = _get_rational_fn(version)
        if version == "D":
            if "random_deviation" in kwargs:
                noise_deviation = kwargs["random_deviation"]
            else:
                noise_deviation = 0.1
            self.register_buffer("noise_deviation", torch.tensor(noise_deviation))
            self.version_kwargs["noise_deviation"] = self.noise_deviation
        elif version == "S":
            if "k" in kwargs:
                k = kwargs["k"]
            else:
                k = 2.0
            self.register_buffer("k", torch.tensor(k))
            self.version_kwargs["k"] = self.k

        self.to(device)

    def forward(self, x):
        return self.activation_function(x, self.numerator, self.denominator, **self.version_kwargs)

    def change_version(self, version):
        if (self.version in ["S", "D"]) or (version in ["S", "D"]):
            raise ValueError(f"Rationals of version 'S' or 'D' can not be changed, got change from version {self.version} to {version}")

        if version == self.version:
            return

        self.activation_function = _get_rational_fn(version)
        self.version = version


class RARE(Rational):
    def __init__(self, name="RARE", group=None, init=None, degrees=(6, 4), device="cpu",
                 train_numerator=True, train_denominator=True, k=2., k_trainable=False, logger=None):
        super().__init__(self, init=init, degrees=degrees, device=device,
                         version="B", train_numerator=train_numerator, train_denominator=train_denominator,
                         name=name, group=group, logger=logger)

        self.k = nn.Parameter(torch.FloatTensor([k]), requires_grad=k_trainable)

    def forward(self, x):
        return self.activation_function(x, self.numerator, self.denominator).mul(torch.relu(x+self.k)).mul(-torch.relu(-x+self.k))


class EmbeddedRational(nn.Module):
    def __init__(self, name="EmbeddedRational", group=None, init=None,
                 degrees=(3, 2), device="cpu", version="A", num_rationals=5,
                 train_numerator=True, train_denominator=True, logger=None, **rat_kwargs):
        super().__init__()

        self.num_rationals = num_rationals
        self.successive_rats = [
            Rational(
                init=init, degrees=degrees, device=device, version=version,
                train_numerator=train_numerator, train_denominator=train_denominator,
                name=f"{name}({i})", group=group, logger=logger, **rat_kwargs,
            ) for i in range(self.num_rationals)
        ]

    def forward(self, x):
        for rat in self.successive_rats:
            x = rat(x)
        return x
