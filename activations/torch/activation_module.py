from dataclasses import dataclass

import torch.linalg
import torch
import torch.nn.functional as F
from activations.utils.find_init_weights import find_weights
from activations.utils.utils import _get_auto_axis_layout, _cleared_arrays
from activations.utils.warnings import RationalImportScipyWarning
from activations.utils.activation_logger import ActivationLogger
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
import io
import PIL.Image
import numpy as np
from termcolor import colored
from random import randint
import torchvision.transforms

import os

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
        self.input_distributions = OrderedDict()
        self.input_labels = OrderedDict()

        # gradient distributions
        self.input_gradient_distributions = OrderedDict()
        self.output_gradient_distributions = OrderedDict()
        self.input_gradient_labels = OrderedDict()
        self.output_gradient_labels = OrderedDict()

        self._grad_handle = None
        self._input_handle = None

    @property
    def groups(self):
        return self._groups
    
    @property
    def use_kde(self):
        return self.display_mode == "kde"
    
    @staticmethod
    def _increment_snapshot_name(name, snapshots):
        while name in snapshots:
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
    
    @torch.no_grad()
    def _call_nohook(self, *args, **kwargs):
        """Runs the ``forward`` method without calling any registered hooks."""
        return self.module.forward(*args, **kwargs)
    
    def save_inputs(self, name, saving=True, max_saves=-1,
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
            hist = NeuronsHistogram(bin_width)
        else:
            hist = Histogram(bin_width)

        name = self._increment_snapshot_name(name, self.input_distributions)
        self.input_distributions[name] = hist
        self.input_labels[name] = label

        self._input_handle = self.module.register_forward_hook(
            _input_hook(
                self, self.input_distributions[name], max_saves,
            )
        )
        

    def save_gradients(self, name, saving=True, max_saves=-1,
                       bin_width="auto", label_in=None, label_out=None):
        if not saving:
            self.logger.warn("Not retrieving gradients anymore")
            self._grad_handle.remove()
            self._grad_handle = None
            return
        
        if self._handle_grads is not None:
            return
        
        name = self._increment_snapshot_name(name, self.input_gradient_distributions)
        self.input_gradient_distributions[name] = Histogram(bin_width)
        self.output_gradient_distributions[name] = Histogram(bin_width)

        self.input_gradient_labels[name] = label_in
        self.output_gradient_labels[name] = label_out

        self._grad_handle = self.register_full_backward_hook(
            _gradient_hook(
                self,
                self.input_gradient_distributions[-1],
                self.output_gradient_distributions[-1],
                max_saves,
            )
        )
            
    def capture(self, name="snapshot_0", other_func=None, label_in=None, label_out=None):
        # self.module.distribution is always None therefore 3rd argument
        # does not influence behaviour
        snapshot = (self.module.state_dict(), other_func, label_in, label_out)
        
        name = self._increment_snapshot_name(name)
        self.snapshots[name] = snapshot

    def input_range(self, name):
        pass # TODO

    def plot_histogram(self, hist, axis, color=None, label=None, tolerance=0.001):
        """Wrapper function for :func:`_plot_histogram`."""
        weights, bins = hist.weights, hist.bins
        kde_fn = lambda n: None
        
        if self.input_retrieval_mode == "neurons":
            if self.display_mode == "kde":
                kde_fn = hist.kde
        else:
            weights, bins = [weights], [bins]
            if self.display_mode == "kde":
                kde_fn = lambda n: hist.kde

        for n, (weights, bins) in enumerate(zip(hist.weights, hist.bins)):
            weights, bins = _cleared_arrays(weights, bins, tolerance=tolerance)
            self._plot_hist(
                weights=weights,
                bins=bins,
                axis=axis,
                kde_fn=kde_fn(n),
                color=color,
                label=label,
            )

    def _plot_histogram(self, weights, bins, axis, kde_fn=None, color=None, label=None):
        """Plots a histogram on a :obj:``plt.Axes``."""
        if kde_fn is None:  # display mode 'bar'
            if len(bins) == len(weights):
                axis.bar(bins, weights/weights.max(),
                         linewidth=0, alpha=0.7, label=label)
            else:
                axis.bar(bins[1:], weights/weights.max(),
                         linewidth=0, alpha=0.7, label=label)
        else:  # display mode 'kde'
            if len(bins) < 5:
                msg = msg = f"Too few bins, maybe reduce bin size. Expected at least 5, got {len(bins)}"
                self.logger.info(msg)
                return

            x = np.linspace(bins[0], bins[-1], 200)
            y = kde_fn(x)
            axis.fill_between(
                x,
                y,
                alpha=0.45,
                color=color,
                label=label
            )

    def show_function(self, x, axis, color,
             name="snapshot_0"):
        current_state = None
        if name is not None:
            current_state = self.module.state_dict()
            state, other_func, label_in, label_out = self.snapshots[name]
            
            self.module.load_state_dict(state)

        y = self._call_nohook(x)
        axis.plot(x, y, color=color, color=self.name)
        for other_func_name in other_func:
            axis.plot(
                x,
                other_func[other_func_name](x),
                label=other_func_name,
            )
        if label_in is not None:
            axis.set_xlabel(label_in)
        if label_out is not None:
            axis.set_ylabel(label_out)

        if current_state is not None:
            self.module.load_state_dict(current_state)


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
        if group is None:
            return tuple(cls._registered_modules.keys())
        
        if not isinstance(group, list):
            group = [group]

        _groups = set(group)
        names = tuple(filter(
            lambda name: not set(cls._registered_modules[name].groups).isdisjoint(_groups),
            cls._registered_modules.keys()
        ))
        return names
    
    @classmethod
    def _get_modules(cls, name=None, group=None, as_dict=False):
        if name is not None and group is not None:
            msg = "Name and group are exclusive"
            raise ValueError(msg)
        
        if group is not None:
            module_names = cls.get_groups(group)
        elif isinstance(name, str):
            module_names = [name]
        else:
            module_names = name

        if as_dict:
            return {name_: cls._registered_modules[name_] for name_ in module_names}
        return [cls._registered_modules[name] for name in module_names]

    @classmethod
    def save_inputs(cls, name=None, group=None, snap_name="snapshot_0", saving=True, auto_stop=False,
                         max_saves=1000, bin_width=0.1, mode=None, input_label=None):
        """Saves inputs of registered modules."""
        for module in cls._get_modules(name=name, group=group):
            module.save_inputs(
                snap_name=snap_name,
                saving=saving,
                max_saves=max_saves if auto_stop else -1,
                bin_width=bin_width,
                mode=mode,
                label=input_label,
            )

    @classmethod
    def save_gradients(cls, name=None, group=None, snap_name="snapshot_0", saving=True, auto_stop=False,
                           max_saves=1000, bin_width="auto", mode=None, input_label=None, output_label=None):
        """Saves gradients of registered modules."""
        for module in cls._get_modules(name=name, group=group):
            module.save_gradients(
                snap_name=snap_name,
                saving=saving,
                max_saves=max_saves if auto_stop else -1,
                bin_width=bin_width,
                mode=mode,
                label_in=input_label,
                label_out=output_label,
            )
        
    @classmethod
    def capture(cls, name=None, group=None, snap_name="snapshot_0",
                other_func=None, returns=False):
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

    @classmethod
    def show_function(cls, name=None, group=None, snap_name=None, x=None, other_func=None,
                      display=False, title=None, axes=None, layout="auto", writer=None,
                      step=None, colors="#1f77b4", x_label=None, y_label=None, ax_title=False,
                      inputs=False, gradients=False, x_mode="expand"):
        """Create figure of multiple modules for one snapshot.
        
        Creates a figure with one subplot for each module. Each module can only be plotted for a single snapshot,
        but snapshots do not have to be equal for all modules.

        Args:
            name (str or list(str), optional):
            group (str or list(str), optional):
            snap_name (str or dict(str, str), optional):
            x (int or tuple or array-like, optional):
            other_func (dict(str, callable)):
            display (bool):
            title (str):
            axes (list(plt.Axes)):
            layout (str or tuple):
            writer ():
            step ():
            colors (str or dict(str, str)):
            x_label, y_label (str or dict(str, str)):
            ax_title (bool):
            x_mode (str): Determines how x axis range is expanded/clipped based
                on given x range (see :param:``x``) and input range. One of ``"expand", "clip"``.
                * ``"expand"``: Always expand to greatest range.
                * ``"clip"``: Always clip to smallest range.
                Defaults to ``"expand"``.
        """
        returns = (axes is None) and not display and (writer is None)
        display = (axes is None) and display and (writer is None)
        tensorboard = (axes is None) and (writer is not None)
        
        modules = cls._get_modules(name, group, as_dict=True)
        n_modules = len(modules)

        if isinstance(snap_name, str):
            snap_name = {name: snap_name for name in modules}
        if isinstance(colors, str):
            colors = {name: colors for name in modules}
        if not isinstance(x, dict):
            x = {name: x for name in modules}
        if not isinstance(x_label, dict):
            x_label = {name: x_label for name in modules}
        if not isinstance(y_label, dict):
            y_label = {name: y_label for name in modules}

        if axes is not None:
            fig = None
            if len(axes) != n_modules:
                msg = f"Expected one axis for each module, got {len(axes)} axes but {n_modules} modules"
                raise ValueError(msg)
        else:
            if layout == "auto":
                layout = _get_auto_axis_layout(n_modules)
            elif len(layout) != 2:
                msg = 'layout should be either "auto" or a tuple of size 2'
                raise ValueError(msg)

            figsize = (layout[1] * 3, layout[0] * 2)
            try:
                import seaborn as sns
                with sns.axes_style("whitegrid"):
                    fig, axes = plt.subplots(*layout, figsize=figsize, squeeze=True)
            except ImportError:
                cls.logger.warn("Could not import seaborn")
                fig, axes = plt.subplots(*layout, figsize=figsize, squeeze=True)
            if title is not None:
                fig.suptitle(title)
        
        if isinstance(axes, plt.Axes):
            axes = {name: axes for name in modules}
        else:
            for ax in axes[n_modules:]:
                ax.remove()
            axes = {name: axes[i] for i, name in enumerate(modules)}

        for name, mod in modules.items():
            x_ = x[name]

            min_x1, max_x1 = None, None
            if (x_ is None) and (not inputs):
                x_ = torch.arange(-3, 3, 0.01, dtype=float)
                min_x1, max_x1 = -3., 3., 600
            if isinstance(x_, int):
                x_ = torch.linspace(-3, 3, x_, dtype=float)
                min_x1, max_x1 = -3., 3.
            elif isinstance(x_, tuple):
                x_ = torch.linspace(*x_, dtype=float)
                min_x1, max_x1 = x_
            elif not isinstance(x_, torch.Tensor):
                x_ = torch.tensor(x_, dtype=float)
                min_x1, max_x1 = torch.min(x_), torch.max(x_)

            min_x2, max_x2 = None, None
            if inputs:
                min_x2, max_x2 = mod.input_range(name=snap_name[name])
                mod.show_inputs()

            if (min_x2 is not None) and (min_x1 is not None):
                if x_mode == "expand":
                    min_x = min(min_x1, min_x2)
                    max_x = max(max_x1, max_x2)
                elif x_mode == "clip":
                    min_x = max(min_x1, min_x2)
                    max_x = min(max_x1, max_x2)
                else:
                    msg = f"Invalid value for `x_mode`, got {x_mode}"
                    raise ValueError(msg)

            mod.show_function(
                x=x_,
                other_func=other_func,
                axis=axes[name],
                color=colors[name],
                name=snap_name[name],
            )

            if other_func is not None:
                for other_func_name in other_func:
                    axes[name].plot(
                        x_,
                        other_func[other_func_name](x_),
                        label=other_func_name,
                    )
            
            axes[name].set_xlim((min_x, max_x))

            if ax_title:
                axes[name].set_title(snap_name)
            if x_label[name] is not None:
                axes[name].set_xlabel(x_label)
            if y_label[name] is not None:
                axes[name].set_ylabel(y_label)

        if fig is not None:
            legend = fig.legend(fancybox=True, shadow=True)
            legend.get_frame().set_alpha(0.4)
            fig.tight_layout()

        if display:
            fig.show()
        elif tensorboard:
            try:
                writer.add_figure(title, fig, step)
            except AttributeError:
                msg = f"Could not use given writer to add figure, got {writer}\n"
                cls.logger.info(msg)
        
        return fig
