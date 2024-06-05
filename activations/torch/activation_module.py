from dataclasses import dataclass
import torch
import torch.nn.functional as F
from activations.utils.find_init_weights import find_weights
from activations.utils.utils import _get_auto_axis_layout, _cleared_arrays
from activations.utils.warnings import RationalImportScipyWarning
from activations.utils.activation_logger import ActivationLogger
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from termcolor import colored
from random import randint

from activations.torch.utils.histograms_numpy import Histogram, NeuronsHistogram



def create_colors(n):
    colors = []
    for i in range(n):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    return colors


def _input_hook(registered_module, histogram, max_saves):
    # by using a list instead of integer the hook can modify ``n_saves``
    # and not just a copy of it. This way it can remove itself when desired.
    n_saves = [0]
    def hook(module, input, output):
        histogram.fill_n(input[0])
        n_saves[0] += 1
        if max_saves > 0 and n_saves[0] >= max_saves:
            registered_module.save_inputs(saving=False)
    return hook


def _gradient_hook(registered_module, histogram_input, histogram_output, max_saves):
    n_saves = [0]
    def hook(module, in_grad, out_grad):
        histogram_input.fill_n(in_grad[0])
        histogram_output.fill_n(out_grad[0])
        n_saves[0] += 1
        if max_saves > 0 and n_saves[0] >= max_saves:
            registered_module.save_gradients(saving=False)
    return hook


class RegisteredModule:
    def __init__(self, name, module, groups, mode, logger):
        self.name = name
        self._groups = groups
        self.module = module
        self.logger = logger

        self.display_mode = mode["dist_display"]

        # function snapshots
        self.snapshots = OrderedDict()

        # input distributions
        self.input_retrieval_mode = mode["irm"]
        self.input_distributions = []
        self.input_labels = []

        # gradient distributions
        self.input_gradient_distributions = []
        self.output_gradient_distributions = []
        self.input_gradient_labels = []
        self.output_gradient_labels = []

        self._grad_handle = None
        self._input_handle = None

    @property
    def groups(self):
        return self._groups
    
    @property
    def use_kde(self):
        return self.display_mode == "kde"
    
    def _increment_snapshot_name(self, name):
        while name in self.snapshots:
            name_ = name.split("_")
            name_[-1] = f"{int(name_[-1]) + 1}"
            name = "_".join(name)

        return name
    
    # needed for compability with Snapshot class
    def numpy(self, *args, **kwargs):
        return self.module.numpy(*args, **kwargs)
    
    # needed for compability with Snapshot class
    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def get_distributions_range(self):
        x_min, x_max = np.inf, -np.inf
        for dist in self.distributions:
            if not dist.is_empty:
                x_min, x_max = min(x_min, dist.range[0]), max(x_max, dist.range[-1])
                size = dist.range[1] - dist.range[0]
        if x_min == np.inf or x_max == np.inf:
            return -3, 3, 0.01
        return x_min, x_max, size
    
    def save_inputs(self, saving=True, max_saves=-1,
                    bin_width=0.1, mode=None, label=None):
        if not saving:
            self.logger.info("Not retrieving input anymore")
            self._input_handle.remove()
            self._input_handle = None
            return
        
        if self._input_handle is not None:  # already retrieving inputs
            return
        
        if mode is None:
            mode = self.input_retrieval_mode
        else:
            self.input_retrieval_mode = mode

        if mode == "neurons":
            self.input_distributions.append(NeuronsHistogram(bin_width))
        else:
            self.input_distributions.append(Histogram(bin_width))
        self.input_labels.append(label)

        self._input_handle = self.module.register_forward_hook(
            _input_hook(
                self, self.input_distributions[-1], max_saves,
            )
        )
        

    def save_gradients(self, saving=True, max_saves=-1,
                       bin_width="auto", label_in=None, label_out=None):
        if not saving:
            self.logger.warn("Not retrieving gradients anymore")
            self._grad_handle.remove()
            self._grad_handle = None
            return
        
        if self._handle_grads is not None:
            return
        
        self.input_gradient_distributions.append(Histogram(bin_width))
        self.output_gradient_distributions.append(Histogram(bin_width))

        self.input_gradient_labels.append(label_in)
        self.output_gradient_labels.append(label_out)

        self._grad_handle = self.register_full_backward_hook(
            _gradient_hook(
                self,
                self.input_gradient_distributions[-1],
                self.output_gradient_distributions[-1],
                max_saves,
            )
        )
            
    def capture(self, name="snapshot_0", other_func=None, returns=False):
        """
        Captures a snapshot of the rational functions and related in the
        snapshot_list variable (or returns it if ``returns=True``).

        Arguments:
                name (str):
                    Name of the snapshot.\n
                    Default ``"snapshot_0"``
                other_funcs (callable):
                    another function to be plotted or a list of other callable
                    functions or a dictionary with the function name as key
                    and the callable as value.
                returns (bool):
                    If ``True``, returns the snapshot.
                    Otherwise, saves it in self.snapshot_list \n
                    Default ``False``
        """
        name = self._increment_snapshot_name(name)

        # self.module.distribution is always None therefore 3rd argument
        # does not influence behaviour
        snapshot = Snapshot(name, self, False, other_func)
        if returns:
            return snapshot
        self.snapshots[name] = snapshot


class ActivationModule:
    _registered_modules = {}  # {module-name: RegisteredModule}
    count = 0
    _step = 0
    use_multiple_axis = False
    distribution_display_mode = "kde"
    histograms_colors = ["red", "green", "black"]
    logger = ActivationLogger(f"ActivationModule")

    @classmethod
    def register(cls, module, name, mode="kde_neurons", group=None, logger=None):
        """Registers a ``torch.nn.Module``. Registered modules can be captured/plotted.
        
        Args:
            module (torch.nn.Module):
            name (str): If name already exists an incrementing integer will be appended.
            mode (str, optional): The mode in which input will be retrieved/plotted. For example
                ``'bar_neurons'`` will create a bar plot for each neuron in layer.
            group (hashable or list of hashables, optional): Group(s) to assign ``module`` to. 
            
        Returns:
            name (str): Name under which module is registered.
        """
        if not isinstance(group, list):
            group = [group]

        if name in cls._registered_modules:
            name = cls._increment_name(f"{name}_0")

        dist_display_mode = "kde"
        if "bar" in mode:
            dist_display_mode = "bar"
        elif "points" in mode:
            dist_display_mode = "points"

        input_retrieval_mode = "neurons"
        if "neurons" not in input_retrieval_mode:
            input_retrieval_mode = "normal"
        
        cls._registered_modules[name] = RegisteredModule(
            name=name,
            module=module,
            groups=group,
            mode={
                "dist_display": dist_display_mode,
                "irm": input_retrieval_mode, 
            },
            logger=cls.logger if logger is None else logger,
        )

        cls.count += 1

        return name

    @classmethod
    def _increment_name(cls, name):
        """Helper method for numerating string.
        
        Args:
            name (str): Must end with ``'_i'`` where ``i`` can be any number. Will Increment ``i`` aslong as modules
                are registered under (incremented) name.
                
        Returns:
            new_name (str): Name for which no other module is registered.
        """
        while name in cls._registered_modules:
            name_ = name.split("_")
            name_[-1] = f"{int(name_[-1])+1}"
            name = "_".join(name_)

        return name

    @classmethod
    def get_groups(cls, group):
        if not isinstance(group, list):
            group = [group]

        # find all modules that belong to given groups
        if group is None:
            names = tuple(cls._registered_modules.keys())
        else:
            _groups = set(group)
            names = tuple(filter(
                lambda name: not set(cls._registered_modules[name].groups).isdisjoint(_groups),
                cls._registered_modules.keys()
            ))

        return names
    
    @classmethod
    def _get_modules(cls, name=None, group=None):
        if name is not None and group is not None:
            msg = "Name and group are exclusive"
            raise ValueError(msg)
        
        if name is None:
            module_names = cls.get_groups(group)
        elif isinstance(name, list):
            module_names = name
        else:
            module_names = [name]

        return [cls._registered_modules[name] for name in module_names]
        
    @classmethod
    def create_snapshot(cls, inputs=False, gradients=False, function=False, timeings=False):
        # create a snapshot including all specified statistics
        pass  # TODO

    @classmethod
    def save_inputs(cls, name=None, group=None, saving=True, auto_stop=False,
                         max_saves=1000, bin_width=0.1, mode=None, save_time=False):
        """Saves inputs of registered modules."""
        for module in cls._get_modules(name=name, group=group):
            module.save_input(
                saving=saving,
                max_saves=max_saves if auto_stop else -1,
                bin_width=bin_width,
                mode=mode,
                save_time=save_time,
            )

    @classmethod
    def save_gradients(cls, name=None, group=None, saving=True, auto_stop=False,
                           max_saves=1000, bin_width="auto", mode=None, save_time=False):
        """Saves gradients of registered modules."""
        for module in cls._get_modules(name=name, group=group):
            module.save_gradient(
                saving=saving,
                max_saves=max_saves if auto_stop else -1,
                bin_width=bin_width,
                mode=mode,
                save_time=save_time,
            )
        
    @classmethod
    def capture(cls, name=None, group=None, snap_name="snapshot_0",
                other_func=None, returns=False):
        """
        Captures a snapshot of every instanciated rational functions and \
        related in the snapshot_list variable (or returns a list of them if \
        ``returns=True``).

        Arguments:
                name (str):
                    Name of the snapshot.\n
                    Default ``"snapshot_0"``
                other_funcs (callable):
                    another function to be plotted or a list of other callable
                    functions or a dictionary with the function name as key
                    and the callable as value.
                returns (bool):
                    If ``True``, returns the snapshot.
                    Otherwise, saves it in self.snapshot_list \n
                    Default ``False``
        """
        modules = cls._get_modules(name=name, group=group)

        if returns:
            captures = {}
            for module in modules:
                captures[module.name] = module.capture(
                    name=snap_name,
                    other_func=other_func,
                    returns=True
                )
            return captures

        for module in modules:
            module.capture(name=snap_name, other_func=other_func, returns=False)
