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

    def show(self, x, other_func, axis, color,
             name="snapshot_0", x_label=None, y_label=None):
        if x is None:
            x = torch.arange(-3, 3, 0.01, dtype=float)
        elif isinstance(x, int):
            x = torch.linspace(-3, 3, x, dtype=float)
        elif isinstance(x, tuple):
            if len(x) == 3:
                x = torch.linspace(*x, dtype=float)
            else:
                x = torch.arange(*x, dtype=float)
        elif isinstance(x, np.array):
            x = torch.tensor(x, dtype=float)
        elif not isinstance(x, torch.Tensor):
            msg = f"Unsupported type for argument `x`, got {x}"
            raise ValueError(msg)

        current_state = None
        if name is not None:
            current_state = self.module.state_dict()
            state, other_func, label_in, label_out = self.snapshots[name]
            
            self.module.load_state_dict(state)

            if x_label is None:
                x_label = label_in
            if y_label is None:
                y_label = label_out

        y = self._call_nohook(x)
        axis.plot(x, y, color=color, color=self.name)
        for other_func_name in other_func:
            axis.plot(
                x,
                other_func[other_func_name](x),
                label=other_func_name,
            )
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)

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
                      step=None, colors="#1f77b4", x_label=None, y_label=None):
        returns = (axes is None) and not display and (writer is None)
        display = (axes is None) and display and (writer is None)
        tensorboard = (axes is None) and (writer is not None)
        ax_title = isinstance(snap_name, dict)
        
        modules = cls._get_modules(name, group, as_dict=True)
        n_modules = len(modules)

        if isinstance(colors, str):
            colors = {name: colors for name in modules}
        if not isinstance(x, dict):
            x = {name: x for name in modules}
        if not isinstance(x_label, dict):
            x_label = {name: x_label for name in modules}
        if not isinstance(y_label, dict):
            y_label = {name: y_label for name in modules}
        if not isinstance(snap_name, dict):
            snap_name = {name: snap_name for name in modules}

        if axes is None:
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
            mod.show(
                x=x[name],
                other_func=other_func,
                axis=axes[name],
                color=colors[name],
                name=snap_name[name],
                x_label=x_label[name],
                y_label=y_label[name],
            )
            if ax_title:
                axes[name].set_title(snap_name)

        if display:
            fig.legend()
            fig.tight_layout()
            fig.show()
        elif tensorboard:
            try:
                writer.add_figure(title, fig, step)
            except AttributeError:
                msg = f"Could not use given writer to add figure, got {writer}\n"
                cls.logger.info(msg)
        elif returns:
            return fig
                     

    @classmethod
    def show(cls, name=None, group=None, snap_name=None, x=None, function=False,
                inputs=False, gradients=False, animated=False, other_func=None,
                display=False, save_to=None, title=None, layout="auto", writer=None,
                step=None, colors="#1f77b4"):
        if not isinstance(snap_name, list):
            snap_name = [snap_name]
        if not isinstance(title, list):
            title = [title] * len(snap_name)

        modules = cls._get_modules(name, group, as_dict=True)

        if layout == "auto":
            layout = _get_auto_axis_layout(len(modules))
        figsize = (layout[1] * 3, layout[0] * 2)

        figs = []
        try:
            import seaborn as sns
            use_seaborn = True
        except ImportError:
            cls.logger.warn("Could not import seaborn")
            use_seaborn = False

        for title_, snapshot in zip(title, snap_name):
            if use_seaborn:
                with sns.axes_style("whitegrid"):
                    fig, axes = plt.subplots(*layout, figsize=figsize, squeeze=True)
            else:
                fig, axes = plt.subplots(*layout, figsize=figsize)
            fig.suptitle(title_)

            if function:
                cls.show_function(
                    name=name,
                    group=group,
                    snap_name=snapshot,
                    x=x,
                    other_func=other_func,
                    display=False,
                    title=None,
                    axes=axes,
                    writer=None,
                    step=None,
                    colors=colors,
                    x_label=None,
                    y_label=None,
                )

            if inputs:
                # TODO
                ...

            fig.legend()
            fig.tight_layout()
            figs.append(fig)
        
        if animated:
            images = []
            buffer = io.BytesIO()

            for fig in figs:
                buffer.seek(0, whence=0)  # overwrite old buffer data
                fig.savefig(buffer, format="png")
                images.append(PIL.Image.open(buffer))

            if save_to is not None:
                images[0].save(save_to, save_all=True, duration=800, loop=0, append_images=images[1:], optimize=False)

            if writer is not None:
                vid = torchvision.transforms.ToTensor(images[0])
                vid_tensor = torch.empty(1, len(images), *vid.shape, dtype=vid.dtype)
                for i, img in enumerate(images[1:]):
                    vid_tensor[0, i+1] = torchvision.transforms.ToTensor(img)

                try:
                    writer.add_video(
                        tag=title,
                        vid_tensor=vid_tensor,
                        global_step=step,
                        fps=1.25  # 800 ms per frame
                    )
                except AttributeError:
                    msg = f"Could not use given writer to add video, got {writer}"
                    cls.logger.info(msg)
            
            if display:
                msg = "Cannot show animation, use save_to or writer argument to store animation."
                cls.logger.info(msg)
        else:  # no animation
            if save_to is not None:
                for i, fig in enumerate(figs):
                    fig.savefig(os.path.join(save_to, f"figure_{i}.png"))

            if writer is not None:
                try:
                    for i, fig in enumerate(figs):
                        writer.add_figure(
                            tag=f"{title[i]}_{i}",
                            figure=fig,
                            global_step=step,
                        )
                except AttributeError:
                    msg = f"Could not use given writer to add figure, got {writer}"
                    cls.logger.info(msg)

                if display:
                    plt.show()

                if (not display) and (writer is None) and (save_to is None):
                    return figs
                else:
                    for fig in figs:
                        plt.close(fig)
