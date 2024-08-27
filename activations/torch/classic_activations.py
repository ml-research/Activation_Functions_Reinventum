import torch.nn as nn
import torch.nn.functional as F
from torch import sin

from activations.torch.activation_module import ActivationModule


class _Base(nn.Module):
    def __init__(self, function, name=None, group=None, logger=None):
        super().__init__()
        self.function = function
        ActivationModule.register(self, name=name, group=group, logger=logger)

    def forward(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class ReLU(_Base):
    """Registered :func:`F.relu` activation.

    Args:
        name, group:
            Parameters used for referencing this module. See :meth:`~activations.torch.activation_module.ActivationModule.register`.

        logger:
            Logger used by this module.
    """

    def __init__(self, name=None, group=None, logger=None):
        super().__init__(F.relu, name, group, logger)


class LReLU(_Base):
    """Registered :func:`F.leaky_relu` activation.

    Args:
        name, group:
            Parameters used for referencing this module. See :meth:`~activations.torch.activation_module.ActivationModule.register`.

        logger:
            Logger used by this module.
    """

    def __init__(self, name=None, group=None, logger=None):
        super().__init__(F.leaky_relu, name, group, logger)


class Tanh(_Base):
    """Registered :func:`F.tanh` activation.

    Args:
        name, group:
            Parameters used for referencing this module. See :meth:`~activations.torch.activation_module.ActivationModule.register`.

        logger:
            Logger used by this module.
    """

    def __init__(self, name=None, group=None, logger=None):
        super().__init__(F.tanh, name, group, logger)


class Sigmoid(_Base):
    """Registered :func:`F.sigmoid` activation.

    Args:
        name, group:
            Parameters used for referencing this module. See :meth:`~activations.torch.activation_module.ActivationModule.register`.

        logger:
            Logger used by this module.
    """

    def __init__(self, name=None, group=None, logger=None):
        super().__init__(F.sigmoid, name, group, logger)


class GLU(_Base):
    """Registered :func:`F.glu` activation.

    Args:
        name, group:
            Parameters used for referencing this module. See :meth:`~activations.torch.activation_module.ActivationModule.register`.

        logger:
            Logger used by this module.
    """

    def __init__(self, name=None, group=None, logger=None):
        super().__init__(F.glu, name, group, logger)


class OneSin(_Base):
    """Registered activation which implements function :math:`f(x)=\\begin{cases}\\text{sin}(x\\cdot\\pi),&x\\in(-1,1)\\\\0,&\\text{otherwise}\\end{cases}`.

    Args:
        name, group:
            Parameters used for referencing this module. See :meth:`~activations.torch.activation_module.ActivationModule.register`.

        logger:
            Logger used by this module.
    """

    def __init__(self, name=None, group=None, logger=None):
        function = (
            lambda x: (x + 1 > 0).float()
            * (x - 1 < 0).float()
            * sin(x * 3.141592653589793)
        )
        super().__init__(function, name, group, logger)
