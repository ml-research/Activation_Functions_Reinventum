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
from activations.torch.learnable_activations.rationals.functions import (
    rational_A,
    rational_B,
    rational_C,
    rational_D,
    rational_nonsafe,
    rational_spline,
    era,
)


_all_versions = ["A", "B", "C", "D", "N", "S", "ERA"]


def _get_rational_fn(version):
    if version not in _all_versions:
        raise ValueError(
            f"Unsupported version, got {version} expected one of {_all_versions}"
        )

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
    elif version == "ERA":
        return era


def find_weights(func, x, degrees, version, **scipyargs):
    """Approximates any function using a rational.

    Args:
        func (callable):
            The target function.

        x (array_like):
            Input data.

        degrees (tuple(int, int)):
            Degrees of polynominals.

        scipyargs:
            Additional arguments passed to :func:`scipy.optimize.cuve_fit` (parameter ``method`` is ``'lm'``).
            Using any of ``f``, ``xdata``, ``ydata``, ``p0``, ``method``, ``full_output`` in ``scipyargs`` will raise
            an Exception.

    Returns:
        w_numerator, w_denominator (list(float)):
            Found weights for numerator/denominator.
    """
    n_num, n_denom = degrees
    if version == "RARE":
        n_num -= 2
    elif version == "C":
        n_denom += 1
    elif version == "ERA":
        assert (
            n_num == n_denom + 1
        ), f"Denominator must have polynominal of one degree smaller than numerator for version 'ERA', got {degrees}"
    n_total = n_num + n_denom

    rat_fn = _get_rational_fn(version)
    if version == "RARE":
        k1 = x - torch.abs(x)
        k2 = x + torch.abs(x)

        def rational(x, *params):
            y = rat_fn(x, torch.tensor(params[:n_num]), torch.tensor(params[n_num:]))
            return y * k1 * k2
    else:

        def rational(x, *params):
            return rat_fn(x, torch.tensor(params[:n_num]), torch.tensor(params[n_num:]))

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
        **scipyargs,
    )[0]

    w_numerator, w_denominator = params[:n_num], params[n_num:]
    return w_numerator.tolist(), w_denominator.tolist()


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

        Args:
            function (callable):
                The function you want to fit to rational.

            x (array):
                The range on which the curves of the functions are fitted
                together.
                Defaults to ``np.arange(-3., 3., 0.1)``.

        Returns:
            ((a, b, c, d), dist):
                The parameters to adjust the function
                (vertical and horizontal scales and bias)
                and the final distance between the rational function and the
                fitted one.
        """
        if x is None:
            x = np.arange(-3.0, 3.0, 0.1)
        (a, b, c, d), distance = find_closest_equivalent(function, x)
        return (a, b, c, d), distance

    def best_fit(self, functions, x=None):
        """
        Compute the distance between the rational and other functions.

        Args:
            functions (list(callable)):
                The other functions.
            x (array):
                The range on which the distance is computed.
                Defaults to ``np.arange(-3., 3., 0.1)``.

        Returns:
            best_func (callable):
                The function with minimal distance to rational.

            best_params (tuple):
                The parameters to adjust the function
                (vertical and horizontal scales and bias).
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
        init (str, callable or None):
            Method of weight initiaization.

            * If of type ``str`` will attempt to load corresponding weights from json.
            * If ``callable`` must have two arguments. The first is the number of weights to return
              and the second is ``True`` for numerator and ``False`` for denominator. Should return
              a ``torch.Tensor``.
            * If ``None`` numerator and denominator will be initialized uniformly in `[0, 1]`.

        degrees (tuple(int, int)):
            The degree of the polynominal for numerator (``degrees[0]``) and denominator (``degrees[1]``).

        device (:class:`torch.device`):
            The device the rational is moved to after initialization.

        version (str):
            The :obj:`version <activations.torch.learnable_activations.rationals.functions>` to use.
            Can be any of ``'A'``, ``'B'``, ``'C'``, ``'D'``, ``'N'``, ``'S'``, ``'ERA'``.

        train_numerator, train_denominator (bool):
            If ``True`` gradients for numerator/denominator will be computed.

        name, group:
            The parameters to identify rational. See parameters ``name``, ``group`` in :meth:`.ActivationModule.register`.

        logger:
            Logger used by rational.

        kwargs:
            Additional parameters specific to version. Allowed args are:

            * ``k`` for version ``S``
            * ``noise_deviation`` for version ``D``.

    Variables:
        numerator, denominator (:class:`torch.nn.Parameter`):
            The parameters used in forward/backward pass.

        degrees (tuple(int, int)):
            The degrees of the two polynominals.

        version (str):
            The currently used version.

        init_approximation (str):
            The initialization used.

        activation_function (callable):
            The actual function which computes the rational.

        version_kwargs (dict):
            Any additional arguments that should be passed to actual function.

        noise_deviation (float):
            If version is ``D`` holds the :func:`deviation <.rational_D>` for the uniform distribution. Otherwise unassigned.

        k (float):
            If version is ``S`` holds the parameter :func:`k <.rational_spline>`. Otherwise attribute unassigned.
    """

    def __init__(
        self,
        init=None,
        degrees=(5, 4),
        device="cpu",
        version="A",
        train_numerator=True,
        train_denominator=True,
        name="Rational",
        group=None,
        logger=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            group=group,
            logger=logger,
        )

        n_num, n_denom = degrees
        if version == "C":
            n_denom += 1
        elif version == "ERA":
            assert (
            n_num == n_denom + 1
        ), f"Denominator must have polynominal of one degree smaller than numerator for version 'ERA', got {degrees}"
            n_num = n_denom + 2

        if isinstance(init, str):
            w_numerator, w_denominator = [
                torch.tensor(weight)
                for weight in JsonHandler.load(version, degrees, init)
            ]
        elif init is None:
            w_numerator = torch.rand(n_num)
            w_denominator = torch.rand(n_denom)
        else:
            w_numerator = init(n_num, True)
            w_denominator = init(n_denom, False)

        self.numerator = nn.Parameter(w_numerator, requires_grad=train_numerator)
        self.denominator = nn.Parameter(w_denominator, requires_grad=train_denominator)

        self.degrees = degrees
        self.version = version
        self.init_approximation = f"{init}"

        self.version_kwargs = {}
        self.activation_function = _get_rational_fn(version)
        if version == "D":
            if "noise_deviation" in kwargs:
                noise_deviation = kwargs["noise_deviation"]
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
        return self.activation_function(
            x, self.numerator, self.denominator, **self.version_kwargs
        )

    def change_version(self, version):
        """Change the version of the rational.

        Args:
            version (str):
                The new version to change to. Versions ``S``, ```D`` can not be change from or to.
        """
        if (self.version in ["S", "D"]) or (version in ["S", "D"]):
            raise ValueError(
                f"Rationals of version 'S' or 'D' can not be changed, got change from version {self.version} to {version}"
            )

        if version == self.version:
            return

        self.activation_function = _get_rational_fn(version)
        self.version = version

    def store(self, name=None):
        """Stores weights in current json file.

        Args:
            name (str):
                Name under which rational should be stored. If ``None`` initialization will be taken.
        """
        if name is None:
            name = self.init_approximation

        JsonHandler.store(
            version=self.version,
            degrees=self.degrees,
            name=name,
            numerator=self.numerator.detach().cpu().tolist(),
            denominator=self.denominator.detach().cpu().tolist(),
        )

    def load(self, name=None):
        """Loads weights from current json file.

        Args:
            name (str):
                Name under which rational is stored. If ``None`` initialization will be taken.
        """
        if name is None:
            name = self.init_approximation

        w_numerator, w_denominator = JsonHandler.load(
            version=self.version,
            degrees=self.degrees,
            name=name,
        )

        device = self.numerator.device
        self.numerator = torch.nn.Parameter(torch.tensor(w_numerator, device=device))
        self.denominator = torch.nn.Parameter(
            torch.tensor(w_denominator, device=device)
        )


class RARE(Rational):
    """RARE as proposed by `Authors <link>`_.

    Args:
        name, group:
            The parameters to identify rational. See parameters ``name``, ``group`` in :meth:`.ActivationModule.register`.

        init, degrees, device, train_numerator, train_denominator, logger:
            See parameters in :class:`.Rational`.

        k (float):
            Offset added before ``ReLU``.

        k_trainable (bool):
            If ``True`` gradients for ``k`` are computed in backward pass.
    """

    def __init__(
        self,
        name="RARE",
        group=None,
        init=None,
        degrees=(6, 4),
        device="cpu",
        train_numerator=True,
        train_denominator=True,
        k=2.0,
        k_trainable=False,
        logger=None,
    ):
        n_num, n_denom = degrees
        super().__init__(
            init=init,
            degrees=(n_num - 2, n_denom),
            device=device,
            version="B",
            train_numerator=train_numerator,
            train_denominator=train_denominator,
            name=name,
            group=group,
            logger=logger,
        )
        self.degrees = degrees

        self.k = nn.Parameter(torch.FloatTensor([k]), requires_grad=k_trainable)

    def forward(self, x):
        return (
            self.activation_function(x, self.numerator, self.denominator)
            .mul(torch.relu(x + self.k))
            .mul(-torch.relu(-x + self.k))
        )


class EmbeddedRational(nn.Module):
    """Multiple chained rationals.

    .. math::
        \\begin{aligned}
            \\text{Let }f_0,...,f_n\\text{ be rationals, }X\\text{ be any input.} \\\\
            y=f_0(f_1(...f_n(X)
        \\end{aligned}
    
    Args:
        name (str):
            The base name of each :class:`.Rational`. The **ith** rational will be registered under ``f'{name}({i})'``.
            
        group, init, degrees, device, version, train_numerator, train_denominator, logger:
            Parameters passed to each :class:`.Rational`.
        
        num_rationals (int):
            Number of rationals.

        rat_kwargs:
            Version specific arguments.
        """

    def __init__(
        self,
        name="EmbeddedRational",
        group=None,
        init=None,
        degrees=(3, 2),
        device="cpu",
        version="A",
        num_rationals=5,
        train_numerator=True,
        train_denominator=True,
        logger=None,
        **rat_kwargs,
    ):
        super().__init__()

        self.num_rationals = num_rationals
        self.successive_rats = [
            Rational(
                init=init,
                degrees=degrees,
                device=device,
                version=version,
                train_numerator=train_numerator,
                train_denominator=train_denominator,
                name=f"{name}({i})",
                group=group,
                logger=logger,
                **rat_kwargs,
            )
            for i in range(self.num_rationals)
        ]

    def forward(self, x):
        for rat in self.successive_rats:
            x = rat(x)
        return x
