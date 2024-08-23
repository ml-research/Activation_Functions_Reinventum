import io
import copy
import itertools
from collections import OrderedDict

import torch
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
import torchvision.transforms

import activations.torch.utils.histogram as histogram
from activations.utils.utils import _get_auto_axis_layout
from activations.utils.activation_logger import ActivationLogger



def _input_hook(registered_module, histogram, max_saves, idx=0):
    # by using a list instead of integer the hook can modify ``n_saves``
    # and not just a copy of it. This way it can remove itself when desired.
    n_saves = [0]
    def hook(module, input, output):
        histogram.fill_n(input[0], ax_idx=idx)
        n_saves[0] += 1
        if max_saves > 0 and n_saves[0] >= max_saves:
            registered_module.save_inputs(name=None, saving=False)
    return hook


def _gradient_hook(registered_module, histogram_input, histogram_output, max_saves):
    n_saves = [0]
    def hook(module, in_grad, out_grad):
        histogram_input.fill_n(in_grad[0], ax_idx=None)
        histogram_output.fill_n(out_grad[0], ax_idx=None)
        n_saves[0] += 1
        if max_saves > 0 and n_saves[0] >= max_saves:
            registered_module.save_gradients(name=None, saving=False)
    return hook


class RegisteredModule:
    """Wrapper for :class:`torch.nn.Module` which handles captureing/plotting of data.
    
    Args:
        name (str):
            Unique ID. Is used for registration in :class:`AcivationModule` and as label.
        
        module (:class:`torch.nn.Module`):
            Wrapped module.
        
        groups (list(hashable)):
            All groups module belongs to.
        
        logger:
            Logger used. 

    Variables:
        name (str):
            Name under which module is registered.

        module (:class:`torch.nn.Module` ):
            Wrapped module.
        
        logger:
            Used logger.

        axis_labels (:class:`OrderedDict` (str, tuple(str, str))):
            Mapping of snapshot name to (x-axis label, y-axis label).

        snapshots (:class:`OrderedDict` (str, tuple)):
            Stores snapshots created by :meth:`RegisteredModule.capture` . Mapping of
            snapshot name to (``state dict``, ``other_func``, ``label``) where
            
            * ``state dict`` is the current state of wrapped module (see :meth:`torch.nn.Module.state_dict`)
            * ``other_func``, ``label`` are the corresponding parameters in :meth:`RegisteredModule.capture`

        input_distributions (:class:`OrderedDict` (str, :class:`~activations.torch.utils.histogram.NeuronsHistogram` )):
            Captured distributions of inputs. Mapping of snapshot name to distribution.

        input_gradient_distributions, output_gradient_distributions (:class:`OrderedDict` (str, :class:`~activations.torch.utils.histogram.Histogram` )):
            Captured distributions of gradients wrt input/output. Mapping of snapshot name to distribution.
    """

    def __init__(self, name, module, groups, logger):
        self.name = name
        self._groups = groups
        self.module = module
        self.logger = logger

        # axis labels
        self.axis_labels = OrderedDict()
        self._current_x_label = None
        self._current_y_label = None

        # function snapshots
        self.snapshots = OrderedDict()

        # input distributions
        self.input_distributions = OrderedDict()

        # gradient distributions
        self.input_gradient_distributions = OrderedDict()
        self.output_gradient_distributions = OrderedDict()

        self._grad_handle = None
        self._input_handle = None

    @property
    def groups(self):
        return self._groups
    
    def set_input_category(self, new_category):
        """Sets x-label for all future snapshots."""
        self._current_x_label = new_category

    def set_output_category(self, new_category):
        """Sets y-label for all future snapshots."""
        self._current_y_label = new_category

    def _update_axis_labels(self, name):
        if name not in self.axis_labels:
            self.axis_labels[name] = (self._current_x_label, self._current_y_label)

    def _has_snapshot(self, name):
        return name in self.axis_labels
    
    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    @torch.no_grad()
    def _call_nohook(self, *args, **kwargs):
        # Calls self.module.forward without calling any registered hooks
        return self.module.forward(*args, **kwargs)
    
    def save_inputs(self, name, saving=True, max_saves=-1,
                    bin_width=None, mode=None, label=None):
        """Save input data as histogram.

        All data is stored by attaching hooks that will be called when module is called.
        
        Args:
            name (str):
                Snapshot name.
            
            saving (bool):
                * If ``True`` start saving inputs.
                * If ``False`` stop saving inputs.
            
            max_saves (int):
                After equal number of calls to :meth:`self.module.forward` stop saving inputs.
            
            bin_width (float):
                Width of bins.
            
            mode (str or int, optional):
                * If ``'layer'``: store all input values in single histogram.
                * otherwise should be the index of axis in input tensor
                  that represents number of neurons in layer.
                
                Defaults to :func:`ActivationModule.default_irm`.
            
            label (str, optional):
                Label for plot legend to use. Defaults to ``'<name>_inputs'``.
        """
        if not saving:
            self.logger.info("Not retrieving input anymore")
            self._input_handle.remove()
            self._input_handle = None
            return
        
        if self._input_handle is not None:  # already retrieving inputs
            self.logger.info("Already retrieving inputs")
            return
        
        if mode is None:
            mode = ActivationModule.default_irm()

        if label is None:
            label = f"{self.name}_inputs"

        if mode == "layer":
            hist = histogram.Histogram(bin_width)
            idx = None
        else:
            hist = histogram.NeuronsHistogram(bin_width)
            idx = mode

        self.input_distributions[name] = (hist, label)
        self._update_axis_labels(name)

        self._input_handle = self.module.register_forward_hook(
            _input_hook(
                self, self.input_distributions[name][0], max_saves, idx=idx,
            )
        )

    def save_gradients(self, name, saving=True, max_saves=-1,
                       bin_width=None, label_in=None, label_out=None):
        """
        Saves gradients as histogram.

        Saves gradients of output wrt. input aswell as upstream gradients in seperate histograms.
        All data is stored by attaching hooks that will be called when tensors ``backward`` is called.
        
        Args:
            name (str):
                Snapshot name.
            
            saving (bool):
                * If ``True`` start saving inputs.
                * If ``False`` stop saving inputs.
            
            max_saves (int):
                After equal number of calls to :meth:`self.module.forward` stop saving inputs.
            
            bin_width (float):
                Width of bins.
            
            label_in, label_out (str, optional):
                Label for plot legend to use. Defaults to ``'<name>_in_grad'``/``'<name>_out_grad'``.
        """
        if not saving:
            self.logger.warn("Not retrieving gradients anymore")
            self._grad_handle.remove()
            self._grad_handle = None
            return
        
        if self._grad_handle is not None:
            self.logger.info("Already retrieving inputs")
            return
        
        if label_in is None:
            label_in = f"{self.name}_in_grad"
        if label_out is None:
            label_out = f"{self.name}_out_grad"
        
        self.input_gradient_distributions[name] = (histogram.Histogram(bin_width), label_in)
        self.output_gradient_distributions[name] = (histogram.Histogram(bin_width), label_out)

        self._update_axis_labels(name)

        self._grad_handle = self.module.register_full_backward_hook(
            _gradient_hook(
                self,
                self.input_gradient_distributions[name][0],
                self.output_gradient_distributions[name][0],
                max_saves,
            )
        )
            
    def capture(self, name, other_func=None, label=None, returns=False):
        """Creates a snapshot of current function.
        
        The snapshot is stored via modules ``state_dict``.
        
        Args:
            name (str):
                Snapshot name.
            
            other_func (dict(str, callable), optional):
                Other functions to include in this snapshot. Keys of ``dict`` will be used as labels.
            
            label (str, optional):
                Plot label used. Defaults to ``'<name>'``
            
            returns (bool):
                If True only return snapshot. In this case snapshot can not be plotted via :class:`ActivationModule`.
        """
        if label is None:
            label = self.name
        snapshot = (copy.deepcopy(self.module.state_dict()), other_func, label)
        
        if returns:
            return snapshot

        self.snapshots[name] = snapshot

        self._update_axis_labels(name)

    def input_range(self, name):
        """Return x-axis range for snapshot."""
        hist, _ = self.input_distributions[name]
        left_edge, right_edge = hist.get_bin_edges()
        return left_edge, right_edge

    def plot_histogram(self, hist, label, axis, tolerance=0.001, use_kde=False, color=None):
        """Plots data from histogram.
        
        Args:
            hist (:class:`~activations.torch.utils.histogram.NeuronsHistogram`):
                Data to plot.
            
            label (str):
                Plot label.
            
            axis (``plt.Axes``):
                Axis to plot on.
            
            tolerance (float):
                All bars with a relative fraction less than ``tolerance`` will be removed before plotting.
            
            use_kde (bool):
                If True plot kde function of histogram.
            
            color (`color <https://matplotlib.org/stable/users/explain/colors/colors.html>`_):
                Color used for plotting.
        """
        if hist.is_empty():
            return

        kde_fn = lambda n: None
        if type(hist) == histogram.NeuronsHistogram:
            weights, bins = hist.weights, hist.bins
            if use_kde:
                kde_fn = hist.kde()
        else:
            weights, bins = [hist.weights], hist.bins
            if use_kde:
                kde_fn = lambda n: hist.kde()

        for n, (w, b) in enumerate(zip(weights, bins)):
            filtered_idxs = histogram.filter_weights(w, tolerance)
            w = w[filtered_idxs]
            b = b[filtered_idxs]

            self._plot_histogram(
                weights=w,
                bins=b,
                bin_size=hist.bin_size,
                axis=axis,
                kde_fn=kde_fn(n),
                label=label,
                color=color,
            )

    def _plot_histogram(self, weights, bins, bin_size, axis, kde_fn=None, label=None, color=None):
        if kde_fn is None:  # display mode 'bar'
            bars = axis.bar(bins, weights, alpha=0.7, label=label, align="edge", width=bin_size, color=color)
            axis.set_ylim(0, max(bars.datavalues))
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
                label=label,
                color=color
            )
            axis.set_ylim(bottom=0.0)

    def show_function(self, x, axis, next_color=None, name="snapshot_0"):
        """Plots snapshot of function.
        
        Args:
            x (any):
                X-Axis values to plot on. Must be accepted by ``forward`` pass of module and ``other_func`` (see :meth:`RegisteredModule.capture`)
            
            axis (:class:`plt.Axes`):
                Axis to plot on.
            
            next_color (callable, optional):
                A function with no arguments which returns a color on call. First for each ``other_func`` a color is used and then for the module.
            
            name (str):
                Name of snapshot to plot.
        """
        if next_color is None:
            next_color = lambda: None

        current_state = None
        label = self.name
        if name is not None:
            current_state = copy.deepcopy(self.module.state_dict())
            state, other_func, label = self.snapshots[name]

            if other_func is not None:
                for other_func_name in other_func:
                    axis.plot(x, other_func(x), label=other_func_name, color=next_color())
            
            self.module.load_state_dict(state)

        y = self._call_nohook(x)
        axis.plot(x, y, label=label, color=next_color())
        
        if name in self.axis_labels:
            x_label, y_label = self.axis_labels[name]
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)

        if current_state is not None:
            self.module.load_state_dict(current_state)

    def show_inputs(self, axis, color=None, name="snapshot_0", tolerance=0.001, use_kde=False):
        """Plots captured input data.
        
        Args:
            axis (``plt.Axes``):
                Axis to plot on.
            
            color (`color <https://matplotlib.org/stable/users/explain/colors/colors.html>`_):
                Color used for Histogram data.
            
            name (str):
                Name of snapshot.
            
            tolerance (float):
                See :meth:`RegisteredModule.plot_histogram`.
            
            use_kde (bool):
                See :meth:`RegisteredModule.plot_histogram`.
        """
        if name is None:
            hist, label = next(reversed(self.input_distributions.values()))
        else:
            hist, label = self.input_distributions[name]

        self.plot_histogram(
            hist=hist,
            label=label,
            axis=axis,
            tolerance=tolerance,
            use_kde=use_kde,
            color=color,
        )

    def show_input_gradients(self, axis, color=None, name="snapshot_0", tolerance=0.001, use_kde=False):
        """Plots captured gradients of output wrt. input.
        
        See :meth:`RegisteredModule.show_inputs` for more info.
        """
        if name is None:
            hist, label = next(reversed(self.input_gradient_distributions.values()))
        else:
            hist, label = self.input_gradient_distributions[name]

        self.plot_histogram(
            hist=hist,
            label=label,
            axis=axis,
            tolerance=tolerance,
            use_kde=use_kde,
            color=color,
        )

    def show_output_gradients(self, axis, color=None, name="snapshot_0", tolerance=0.001, use_kde=False):
        """Plots captured upstream gradients.
        
        See :meth:`RegisteredModule.show_inputs` for more info.
        """
        if name is None:
            hist, label = next(reversed(self.output_gradient_distributions.values()))
        else:
            hist, label = self.output_gradient_distributions[name]

        self.plot_histogram(
            hist=hist,
            label=label,
            axis=axis,
            tolerance=tolerance,
            use_kde=use_kde,
            color=color,
        )
        


class ActivationModule:
    """Static class which provides functionality to capture/plot data of multiple :class:`torch.nn.Module`."""

    _registered_modules = {}  # {module-name: RegisteredModule}
    _snapshot_names = []
    _count = 0
    _step = 0
    _logger = ActivationLogger(f"ActivationModule")
    _plotting_style = {}
    _default_irm = "layer"
    _color_cycle = None

    @classmethod
    def set_global_color_cycle(cls, colors):
        """Set a cycle of colors used across axes.
        
        Each time smething is plotted the next color drawn. The cycle is not reset for each axis. 

        Args:
            colors (iterable):
                Colors to endlessly cycle through. If ``None`` will remove current
                cycle and use ``matplotlb.rc_params['prop_cycle']`` for each axis.
        """
        if colors is None:
            cls._color_cycle = None
            return

        cls._color_cycle = itertools.cycle(colors)

    @classmethod
    def _get_next_color(cls):
        if cls._color_cycle is None:
            return None
        
        return next(cls._color_cycle)

    @classmethod
    def default_irm(cls, irm=None):
        """Setter/Getter for the default input retrieval mode (**irm**).
        
        Args:
            irm (str or int, optional):
                See :meth:`RegisteredModule.capture` for detailed information about **irm**. If omitted will return current **irm**.
        """
        if irm is None:
            return cls._default_irm
        
        if (not isinstance(irm, int)) and (not irm == "layer"):
            raise ValueError(f"Unsupported irm, got {irm}")
        cls._default_irm = irm

    @classmethod
    def set_logger(cls, logger):
        """Set default logger.
        
        Does not change assigned loggers for already registered modules.
        """
        cls._logger = logger

    @classmethod
    def get_plotting_style(cls):
        """Get current ``rc_params`` modifications."""
        return {}.update(cls._plotting_style)  # copy to prevent modifications
    
    @classmethod
    def modules(cls):
        """Get list of names for registered modules."""
        return list(cls._registered_modules.keys())

    @classmethod
    def set_plotting_style(cls, path=None, rc_params=None, update=False):
        """Set the used ``matplotlib`` style.
         
        This method will only modify the ``rc_params`` in local scope.
         
        Args:
            path (str or pathlike, optional):
                Path to file containing ``rcparam, value`` pairs (seperated by colon) on each line.
            
            rc_params (dict(str, str), optional):
                ``rcparams`` to use. If a param is used in ``path`` and ``rc_params`` will use value of ``rc_params``.
            
            update (bool):
                If ``True``, only update existing `rc_params` instead of overwriting.
        """
        params = {}
        if path is not None:
            with open(path, "rt") as f:
                for line in f.readlines():
                    key, value = line.split(":")
                    params[key.strip(" ")] = value.strip(" ").strip("\n")

        if rc_params is not None:
            params.update(rc_params)

        if update:
            cls._plotting_style.update(params)
        else:
            cls._plotting_style = params

    @classmethod
    def register(cls, module, name, group=None, logger=None):
        """Registers a :class:`torch.nn.Module`. Registered modules can be captured/plotted.
        
        Args:
            module (:class:`torch.nn.Module`):
                The module to register.
            
            name (str):
                Name under which module will be registered. If name already exists an incrementing integer will be appended.
            
            group (hashable or list of hashables, optional):
                Group(s) to assign ``module`` to. 
            
        Returns:
            name (str):
                Name under which module is registered.
        """
        if not isinstance(group, list):
            group = [group]

        if name in cls._registered_modules:
            if not name.split("_")[-1].isdigit():
                name = f"{name}_0"
            name = cls._increment_name(name, tuple(cls._registered_modules.keys()))
        
        cls._registered_modules[name] = RegisteredModule(
            name=name,
            module=module,
            groups=group,
            logger=cls._logger if logger is None else logger,
        )

        cls._count += 1

        return name

    @classmethod
    def _increment_name(cls, name, blocked_names):
        """Helper method for enumerating a string.
        
        Args:
            name (str):
                Must end with ``'_i'`` where ``i`` can be any number. Will Increment ``i`` aslong as modules
                are registered under (incremented) name.
                
        Returns:
            new_name (str):
                Name for which no other module is registered.
        """
        while name in blocked_names:
            name_ = name.split("_")
            name_[-1] = f"{int(name_[-1])+1}"
            name = "_".join(name_)

        return name

    @classmethod
    def get_groups(cls, group):
        """Get names of modules belonging to specific groups.
        
        Args:
            group (str or list(str)):
                * If ``str``: return all module names registered in this group.
                * If ``list(str)``: return all module names registered in at least one group.
                * If ``None``: return all registered modules.
                
        Returns:
            names (tuple):
                All registered modules matching given group(s).
        """
        if group is None:
            return tuple(cls._registered_modules.keys())
        
        if not isinstance(group, list):
            group = [group]

        _groups = set(group)
        names = tuple(
            filter(
                lambda name: not set(cls._registered_modules[name].groups).isdisjoint(_groups),
                cls._registered_modules.keys()
            )
        )
        return names
    
    @classmethod
    def _get_modules(cls, name=None, group=None):
        """Get all modules based on given name(s) and group(s).
        
        Args:
            name, group (str or list(str)):
                Mutually exclusive parameters to filter modules.
                If both are ``None`` all modules will be returned.
        Returns:
            modules (list(:class:`RegisteredModule`)):
                Found modules.
        """
        
        if (name is not None) and (group is not None):
            msg = "Name and group are exclusive"
            raise ValueError(msg)
        
        if group is not None:
            module_names = cls.get_groups(group)
        elif name is None:
            module_names = cls._registered_modules.keys()
        elif isinstance(name, str):
            module_names = [name]
        else:
            module_names = name

        return [cls._registered_modules[name] for name in module_names]
    
    @classmethod
    def set_labels(cls, name=None, group=None, x_label=None, y_label=None):
        """Sets x/y labels for plot axes.
        
        Args:
            x_label, y_label (str, optional):
                Label for x/y-axis used in plots. If one is given (other is ``None``) changes only the given label.
                If both are ``None`` removes current labels.

        .. note::
            Labels are stored in snapshots therefore changes to labels are only taken into account
            by future calls to :meth:`ActivationModule.create_snapshot`. Changes to existing snapshot 
            labels must be done manually (see :attr:`RegisteredModule.axis_labels`).
        """
        modules = cls._get_modules(name=name, group=group)

        if (x_label is None) and (y_label is None):
            for mod in modules:
                mod.set_input_category(None)
                mod.set_output_category(None)
            return

        for mod in modules:
            if x_label is not None:
                mod.set_input_category(x_label)
            if y_label is not None:
                mod.set_output_category(y_label)
    
    @classmethod
    def get_snapshots(cls, name=None, group=None):
        """Return all shared snapshots in order.

        A snapshot is shared if all modules have a snapshot of equal name.

        Returns:
            snapshots (list(str)):
                Snapshots that are common to all given modules.
        """
        modules = cls._get_modules(name=name, group=group)

        snapshots = [
            snapshot for snapshot in cls._snapshot_names
            if len(filter(lambda x: x._has_snapshot(snapshot), modules)) == len(modules)
        ]

        return snapshots

    @classmethod
    def save_inputs(cls, name=None, group=None, snap_name="snapshot_0", saving=True,
                         max_saves=1000, bin_width=None, mode=None, label_in=None):
        """Saves inputs of registered modules.
        
        For detailed information see :meth:`ActivationModule.create_snapshot`.

        .. note::
            :meth:`ActivationModule.create_snapshot` should be used for creating snapshots. Otherwise existing snapshots may be overwritten.
        """
        modules = cls._get_modules(name=name, group=group)

        if not isinstance(label_in, dict):
            label_in = {module.name: label_in for module in modules}

        for module in modules:
            module.save_inputs(
                name=snap_name,
                saving=saving,
                max_saves=-1 if max_saves is None else max_saves,
                bin_width=bin_width,
                mode=mode,
                label=label_in[module.name],
            )

    @classmethod
    def save_gradients(cls, name=None, group=None, snap_name="snapshot_0", saving=True,
                           max_saves=1000, bin_width=None, label_grad_in=None, label_grad_out=None):
        """Saves gradients of registered modules.
        
        For detailed information see :meth:`ActivationModule.create_snapshot`.

        .. note::
            :meth:`ActivationModule.create_snapshot` should be used for creating snapshots. Otherwise existing snapshots may be overwritten.
        """
        modules = cls._get_modules(name=name, group=group)

        if not isinstance(label_grad_in, dict):
            label_grad_in = {module.name: label_grad_in for module in modules}
        if not isinstance(label_grad_out, dict):
            label_grad_out = {module.name: label_grad_out for module in modules}

        for module in modules:
            module.save_gradients(
                name=snap_name,
                saving=saving,
                max_saves=-1 if max_saves is None else max_saves,
                bin_width=bin_width,
                label_in=label_grad_in[module.name],
                label_out=label_grad_out[module.name],
            )
        
    @classmethod
    def capture(cls, name=None, group=None, snap_name="snapshot_0",
                other_func=None, returns=False):
        """Saves snapshot of registered modules

        Args:
            returns (bool):
                If ``True`` snapshots are returned and not saved.
        
        For detailed information of parameters see :meth:`ActivationModule.create_snapshot`.

        .. note::
            :meth:`ActivationModule.create_snapshot` should be used for creating snapshots. Otherwise existing snapshots may be overwritten.
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

    @classmethod
    def create_snapshot(cls, name=None, group=None, snap_name="snapshot_0",
                        other_func=None, max_saves=1000, bin_width=None, label_in=None,
                        label_grad_in=None, label_grad_out=None, irm="layer",
                        function=False, inputs=False, gradients=False):
        """Create snapshots for later plotting.
        
        Args:
            name, group (str or list(str)):
                Mutually exclusive parameters to specify all registered modules for which a snapshot is created.
            
            snap_name (str):
                Name of new snapshot. If name is already existing an increasing integer will be suffixed.
            
            other_func (dict(str, callable)):
                Other functions to include in plots (the keys will be used as labels).
            
            max_saves (int):
                Maximum number of times inputs/gradients are saved. After the maximum is exceeded hooks will automatically remove themselfs.
            
            bin_width (float):
                Width of histogram bins.
            
            label_in, label_grad_in, label_grad_out (str or dict(str, str)):
                Labels for inputs/gradients. If given as dict should have a key matching each module name.
            
            irm (str or int):
                Input retrieval mode. See :meth:`RegisteredModule.capture` for more information.
            
            function, inputs, gradients (bool):
                Data to be stored in snapshot.
        """
        snap_name = cls._increment_name(snap_name, cls._snapshot_names)

        if function:
            cls.capture(
                name=name,
                group=group,
                snap_name=snap_name,
                other_func=other_func,
                returns=False,
            )
        if inputs:
            cls.save_inputs(
                name=name,
                group=group,
                snap_name=snap_name,
                saving=True,
                max_saves=max_saves,
                bin_width=bin_width,
                mode=irm,
                label_in=label_in,
            )
        if gradients:
            cls.save_gradients(
                name=name,
                group=group,
                snap_name=snap_name,
                saving=True,
                max_saves=max_saves,
                bin_width=bin_width,
                label_grad_in=label_grad_in,
                label_grad_out=label_grad_out,
            )

    @classmethod
    def stop_saving(cls, name=None, group=None, inputs=True, gradients=True):
        """Stop retrieving inputs/gradients.
        
        Removes all attached forward/backward hooks.
        
        Args:
            name, group (str or list(str)):
                Mutually exclusive parameters for specifying modules.
            inputs (bool):
                If ``True`` stop retrieving inputs.
            gradients (bool):
                If ``True`` stop retrieving gradients.
        """
        if inputs:
            cls.save_inputs(
                name=name,
                group=group,
                saving=False,
            )

        if gradients:
            cls.save_gradients(
                name=name,
                group=group,
                saving=False,
            )

    @classmethod
    def show_function(cls, name=None, group=None, snap_name=None, x=None, other_func=None,
                      display=False, title=None, axes=None, layout="auto", writer=None,
                      step=None, x_label=None, y_label=None, ax_title=True, function=False,
                      inputs=False, gradients_input=False, gradients_output=False, x_mode="expand",
                      tol_in=0.001, tol_grad_in=0.001, tol_grad_out=0.001, save_to=None, use_kde=False,
                      legend=True, **fig_kw):
        """Create figure of multiple modules for single snapshot.
        
        Creates a figure with one axis for each module. Each module can only be plotted for a single snapshot,
        but snapshots do not have to be equal for all modules.

        Args:
            name, group (str or list(str)):
                Mutually exclusive parameters specifying modules.
            
            snap_name (str or dict(str, str)):
                Snapshot to be used.
            
            x (int or tuple(float, float, int) or array-like):
                The x-values to plot snapshot on. Defaults to ``torch.linspace(-3., 3., 100)``.
                    * If ``int`` number of points in interval [-3, 3].
                    * If ``tuple(float, float, int)`` specifies min, max value and number of points.
                    * If ``array-like`` Concrete values to use.
            
            other_func (dict(str, callable)):
                Other functions to use. Keys are used as plot labels.
            
            display (bool):
                If ``True`` shows plot using ``plt.show`` (blocking until figure is closed).
            
            title (str):
                Title of figure. Is also used as tag for SummaryWriter.
            
            axes (``plt.Axes`` or list(``plt.Axes``)):
                Axes to plot on.
                    * If ``plt.Axes`` plot all snapshots on single axis.
                    * If ``list(plt.Axes)``: plots each module on own axis.
                
            layout ("auto" or tuple(n_rows, n_cols)):
                Determines the `subplot layout`_. If ``layout=="auto"``
                the number of rows and columns is determined automatically. Otherwise ``n_rows*n_cols``
                should be greater than number of modules to plot or equal to 1, in which case all modules
                will be plotted in single plot.
            
            writer (:class:`torch.utils.tensorboard.SummaryWriter`):
                Add figure on SummaryWriter. Will add before showing if ``display==True``.
            
            step (int):
                Global step for ``writer``.
            
            x_label, y_label (str or dict(str, str)):
                Label for x/y-axes.
            
            ax_title (bool):
                If ``True`` each axis uses the module name as title. Otherwise axis have no title.
            
            x_mode (str):
                Determines how x axis limits are choosen between input histogram range and given parameter ``x``. Defaults to ``'expand'``.
                    * If ``'expand'`` expand to greatest range.
                    * If ``'clip'`` clip to smallest range.
                    * If ``'x'`` use only parameter ``x``.
            
            tol_in, tol_grad_in, tol_grad_out (float):
                Tolerance values for input/gradient histograms (see :meth:`RegisteredModule.plot_histogram`)
            
            save_to (str or pathlike):
                Path to save figure to. File extension must be supported by matplotlib.
            
            use_kde (bool):
                Plot kde function instead of barplots for all histograms.
            
            legend (bool):
                Create legend for plot.
            
            fig_kw:
                Keyword arguments passed to :meth:`matplotlib.pyplot.plt.subplots`. Ignored if ``axes`` is given.

        .. note::
            If a parameter is allowed to be passed as :class:`dict` it can be specified per module by using module names as keys.
            In this case the :class:`dict` must have a key matching each module name.


        .. _subplot layout:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
        """
        if x_mode not in ["expand", "clip", "x"]:
            msg = f"Invalid x_mode, got {x_mode}"
            raise ValueError(msg)
        if (not inputs):
            x_mode = "x"

        modules = cls._get_modules(name, group)
        n_modules = len(modules)

        if not isinstance(snap_name, dict):
            snap_name = {module.name: snap_name for module in modules}
        if not isinstance(x, dict):
            x = {module.name: x for module in modules}
        if not isinstance(x_label, dict):
            x_label = {module.name: x_label for module in modules}
        if not isinstance(y_label, dict):
            y_label = {module.name: y_label for module in modules}
        
        if axes is not None:
            fig = None
            if not isinstance(axes, plt.Axes) and (len(axes) != n_modules):
                msg = f"Expected one axis for each module, got {len(axes)} axes but {n_modules} modules"
                raise ValueError(msg)
        else:
            if layout == "auto":
                layout = _get_auto_axis_layout(n_modules)
            elif len(layout) != 2:
                msg = 'layout should be either "auto", "together" or a tuple of size 2'
                raise ValueError(msg)

            figsize = (layout[1] * 6, layout[0] * 4)
            with plt.rc_context(rc=cls._plotting_style):
                fig, axes = plt.subplots(*layout, figsize=figsize, squeeze=True, **fig_kw)
            if title is not None:
                fig.suptitle(title)
        
        if isinstance(axes, plt.Axes):
            axes = {module.name: axes for module in modules}
        else:
            axes = axes.flatten()
            for ax in axes[n_modules:]:
                ax.remove()
            axes = {module.name: axes[i] for i, module in enumerate(modules)}

        for module in modules:
            x_ = x[module.name]
            axis = axes[module.name]
            if inputs or gradients_input or gradients_output:
                twin_x = axis.twinx()

            min_x1, max_x1 = None, None
            if x_ is None:
                x_ = torch.linspace(-3., 3., 100)
                min_x1, max_x1 = -3., 3.
            elif isinstance(x_, int):
                x_ = torch.linspace(-3, 3, x_)
                min_x1, max_x1 = -3., 3.
            elif isinstance(x_, tuple):
                min_x1, max_x1 = x_[0], x_[1]
                x_ = torch.linspace(*x_)
            elif not isinstance(x_, torch.Tensor):
                x_ = torch.tensor(x_)
                min_x1, max_x1 = torch.min(x_), torch.max(x_)

            min_x2, max_x2 = None, None
            if inputs:
                min_x2, max_x2 = module.input_range(name=snap_name[module.name])
                module.show_inputs(
                    axis=twin_x,
                    name=snap_name[module.name],
                    tolerance=tol_in,
                    use_kde=use_kde,
                    color=cls._get_next_color(),
                )

            if gradients_input:
                module.show_input_gradients(
                    axis=twin_x,
                    name=snap_name[module.name],
                    tolerance=tol_grad_in,
                    use_kde=use_kde,
                    color=cls._get_next_color(),
                )
            if gradients_output:
                module.show_output_gradients(
                    axis=twin_x,
                    name=snap_name[module.name],
                    tolerance=tol_grad_out,
                    use_kde=use_kde,
                    color=cls._get_next_color(),
                )    

            if (x_mode == "x") or (min_x2 is None):
                min_x = min_x1
                max_x = max_x1
            elif (min_x2 is not None) and (min_x1 is not None):
                if x_mode == "expand":
                    min_x = min(min_x1, min_x2)
                    max_x = max(max_x1, max_x2)
                elif x_mode == "clip":
                    min_x = max(min_x1, min_x2)
                    max_x = min(max_x1, max_x2)
            elif min_x1 is None:
                min_x = min_x2
                max_x = max_x2

            if function:
                module.show_function(
                    x=x_,
                    axis=axis,
                    name=snap_name[module.name],
                    next_color=cls._get_next_color,
                )

            if other_func is not None:
                for other_func_name in other_func:
                    axis.plot(
                        x_,
                        other_func[other_func_name](x_),
                        label=other_func_name,
                        color=cls._get_next_color(),
                    )
            
            axis.set_xlim((min_x, max_x))

            if ax_title:
                axis.set_title(module.name)
            if x_label[module.name] is not None:
                axis.set_xlabel(x_label)
            if y_label[module.name] is not None:
                axis.set_ylabel(y_label)

        if fig is not None:
            if legend:
                legend = fig.legend(fancybox=True, shadow=True)
                legend.get_frame().set_alpha(0.4)
            fig.tight_layout()

            if save_to is not None:
                fig.savefig(save_to, dpi="figure")
            if writer is not None:
                try:
                    writer.add_figure(title, fig, step)
                except AttributeError:
                    msg = f"Could not use given writer to add figure, got {writer}\n"
                    cls._logger.info(msg)
            if display:
                fig.show()

        return fig
    
    @classmethod
    def export_evolution_graphs(cls, path, name=None, group=None, snap_names=None, layout="auto", video_writer=None, step=None, tag=None,
                                function=True, inputs=False, gradients_input=False, gradients_output=False, legend=False, **kwargs):
        """Creates animation over multiple snapshots.
        
        Args:
            path (str or pathlike):
                Path to save animation to.

            snap_names (list(str) or list(dict(str, str))):
                All snapshots to animate over.
                Each element should be accepted by :meth:`ActivationModule.show_function`.
            
            video_writer (:class:`torch.utils.tensorboard.SummaryWriter`):
                Writer to add animation. If writer does not support videos
                only a log message is produced. 
            
            tag (str):
                Tag for ``video_writer``.
            
            kwargs:
                For a full list of additional arguments see :meth:`ActivationModule.show_function`.

        Raises:
            TypeError:
                If keyword argument ``axes`` is passed.
        """
        if len(snap_names) == 1:
            msg = "At least 2 snapshots must be given, got 1"
            raise ValueError(msg)

        n_modules = len(cls._get_modules(name=name, group=group))
        if layout == "auto":
            layout = _get_auto_axis_layout(n_modules)

        figsize = (layout[1] * 6, layout[0] * 4)

        images = []
        for snap_name in snap_names:
            with plt.rc_context(rc=cls._plotting_style):
                fig, axes = plt.subplots(*layout, figsize=figsize, squeeze=True)
            cls.show_function(
                name=name,
                group=group,
                layout=None,
                axes=axes,
                snap_name=snap_name,
                step=step,
                function=function,
                inputs=inputs,
                gradients_input=gradients_input,
                gradients_output=gradients_output,
                **kwargs
            )

            buffer = io.BytesIO()
            fig.suptitle(snap_name)
            if legend:
                fig.legend()
            fig.tight_layout()
            fig.savefig(buffer, format="png")
            buffer.seek(0, 0)
            images.append(PIL.Image.open(buffer))
            plt.close(fig)

        images[0].save(path, save_all=True, duration=800, loop=0, append_images=images[1:], format="GIF")

        if video_writer is not None:
            vid = torchvision.transforms.ToTensor(images[0])
            vid_tensor = torch.empty(1, len(images), *vid.shape, dtype=vid.dtype)
            for i, img in enumerate(images[1:]):
                vid_tensor[0, i+1] = torchvision.transforms.ToTensor(img)

            try:
                video_writer.add_video(
                    tag=tag,
                    vid_tensor=vid_tensor,
                    global_step=step,
                    fps=1.25  # 800 ms per frame
                )
            except AttributeError:
                msg = f"Could not use given writer to add video, got {video_writer}"
                cls._logger.info(msg)