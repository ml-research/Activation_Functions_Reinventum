import io
import copy
from collections import OrderedDict

import torch
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
import torchvision.transforms

import activations.torch.utils.histogram as histogram
from activations.utils.utils import _get_auto_axis_layout
from activations.utils.activation_logger import ActivationLogger



def _input_hook(registered_module, histogram, max_saves):
    # by using a list instead of integer the hook can modify ``n_saves``
    # and not just a copy of it. This way it can remove itself when desired.
    n_saves = [0]
    def hook(module, input, output):
        histogram.fill_n(input[0])
        n_saves[0] += 1
        if max_saves > 0 and n_saves[0] >= max_saves:
            registered_module.save_inputs(name=None, saving=False)
    return hook


def _gradient_hook(registered_module, histogram_input, histogram_output, max_saves):
    n_saves = [0]
    def hook(module, in_grad, out_grad):
        histogram_input.fill_n(in_grad[0])
        histogram_output.fill_n(out_grad[0])
        n_saves[0] += 1
        if max_saves > 0 and n_saves[0] >= max_saves:
            registered_module.save_gradients(name=None, saving=False)
    return hook


class RegisteredModule:
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
        self._current_x_label = new_category

    def set_output_category(self, new_category):
        self._current_y_label = new_category

    def _update_axis_labels(self, name):
        if name not in self.axis_labels:
            self.axis_labels[name] = (self._current_x_label, self._current_y_label)

    def _has_snapshot(self, name):
        return name in self.axis_labels
    
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
                    bin_width=None, mode=None, label=None):
        if not saving:
            self.logger.info("Not retrieving input anymore")
            self._input_handle.remove()
            self._input_handle = None
            return
        
        if self._input_handle is not None:  # already retrieving inputs
            self.logger.info("Already retrieving inputs")
            return
        
        if mode is None:
            mode = ActivationModule._default_irm

        if label is None:
            label = f"{self.name}_inputs"

        if mode == "neurons":
            hist = histogram.NeuronsHistogram(bin_width)
        else:
            hist = histogram.Histogram(bin_width)

        self.input_distributions[name] = (hist, label)
        self._update_axis_labels(name)

        self._input_handle = self.module.register_forward_hook(
            _input_hook(
                self, self.input_distributions[name][0], max_saves,
            )
        )

    def save_gradients(self, name, saving=True, max_saves=-1,
                       bin_width=None, label_in=None, label_out=None):
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
            
    def capture(self, name="snapshot_0", other_func=None, label=None, returns=False):
        # self.module.distribution is always None therefore 3rd argument
        # does not influence behaviour
        if label is None:
            label = self.name
        snapshot = (copy.deepcopy(self.module.state_dict()), other_func, label)
        
        if returns:
            return snapshot

        self.snapshots[name] = snapshot

        self._update_axis_labels(name)

    def input_range(self, name):
        hist, _ = self.input_distributions[name]
        left_edge, right_edge = hist.get_bin_edges()
        return left_edge, right_edge

    def plot_histogram(self, name, histograms, axis, tolerance=0.001, use_kde=False):
        if name is None:
            hist, label = next(reversed(histograms.values()))  # last element
        else:
            hist, label = histograms[name]

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
            )

    def _plot_histogram(self, weights, bins, bin_size, axis, kde_fn=None, label=None):
        """Plots a histogram on a :obj:``plt.Axes``."""
        if kde_fn is None:  # display mode 'bar'
            axis.bar(bins, weights, linewidth=0,
                     alpha=0.7, label=label, align="edge", width=bin_size)
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
                label=label
            )

    def show_function(self, x, axis, name="snapshot_0"):
        current_state = None
        label = self.name
        if name is not None:
            current_state = copy.deepcopy(self.module.state_dict())
            state, other_func, label = self.snapshots[name]

            if other_func is not None:
                for other_func_name in other_func:
                    axis.plot(x, other_func(x), label=other_func_name)
            
            self.module.load_state_dict(state)

        y = self._call_nohook(x)
        axis.plot(x, y, label=label)
        
        if name in self.axis_labels:
            x_label, y_label = self.axis_labels[name]
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)

        if current_state is not None:
            self.module.load_state_dict(current_state)

    def show_inputs(self, axis, color=None, name="snapshot_0", tolerance=0.001, use_kde=False):
        self.plot_histogram(
            name=name,
            histograms=self.input_distributions,
            axis=axis,
            color=color,
            tolerance=tolerance,
            use_kde=use_kde,
        )

    def show_input_gradients(self, axis, color=None, name="snapshot_0", tolerance=0.001, use_kde=False):
        self.plot_histogram(
            name=name,
            histograms=self.input_gradient_distributions,
            axis=axis,
            color=color,
            tolerance=tolerance,
            use_kde=use_kde,
        )

    def show_output_gradients(self, axis, color=None, name="snapshot_0", tolerance=0.001, use_kde=False):
        self.plot_histogram(
            name=name,
            histograms=self.output_gradient_distributions,
            axis=axis,
            color=color,
            tolerance=tolerance,
            use_kde=use_kde
        )


class ActivationModule:
    _registered_modules = {}  # {module-name: RegisteredModule}
    _snapshot_names = []
    _count = 0
    _step = 0
    _logger = ActivationLogger(f"ActivationModule")
    _plotting_style = {}
    _default_irm = "layer"

    @classmethod
    def default_irm(cls, irm=None):
        if irm is None:
            return cls._default_irm
        
        if not irm in ["layer", "neurons"]:
            raise ValueError(f"Unsupported irm, got {irm}")
        cls._default_irm = irm

    @classmethod
    def set_logger(cls, logger):
        cls._logger = logger

    @classmethod
    def get_plotting_style(cls):
        return {}.update(cls._plotting_style)  # copy to prevent modifications

    @classmethod
    def set_plotting_style(cls, path=None, rc_params=None, update=False):
        """Set the used ``matplotlib`` style.
         
        This method will only modify the `rc_params` in local scope.
         
        Args:
            path (str or pathlike, optional): Path to file containing ``rcparam, value`` pairs (seperated by colon) on each line.
            rc_params (dict(str, str), optional): ``rcparams`` to use. Will overwrite existing ``rcparams`` in ``path``.
            update (bool): If ``True``, only update existing `rc_params`.
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
        """Registers a ``torch.nn.Module``. Registered modules can be captured/plotted.
        
        Args:
            module (torch.nn.Module):
            name (str): If name already exists an incrementing integer will be appended.
            group (hashable or list of hashables, optional): Group(s) to assign ``module`` to. 
            
        Returns:
            name (str): Name under which module is registered.
        """
        if not isinstance(group, list):
            group = [group]

        if name in cls._registered_modules:
            name = cls._increment_name(f"{name}_0", tuple(cls._registered_modules.keys()))
        
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
        """Helper method for numerating string.
        
        Args:
            name (str): Must end with ``'_i'`` where ``i`` can be any number. Will Increment ``i`` aslong as modules
                are registered under (incremented) name.
                
        Returns:
            new_name (str): Name for which no other module is registered.
        """
        while name in blocked_names:
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
        names = tuple(
            filter(
                lambda name: not set(cls._registered_modules[name].groups).isdisjoint(_groups),
                cls._registered_modules.keys()
            )
        )
        return names
    
    @classmethod
    def _get_modules(cls, name=None, group=None):
        if name is not None and group is not None:
            msg = "Name and group are exclusive"
            raise ValueError(msg)
        
        if group is not None:
            module_names = cls.get_groups(group)
        elif isinstance(name, str):
            module_names = [name]
        else:
            module_names = name

        return [cls._registered_modules[name] for name in module_names]
    
    @classmethod
    def get_snapshots(cls, name=None, group=None):
        """Return all shared snapshots in order.

        Returns:
            snapshots (list(str)): Snapshots that are common to all given modules.
        """
        modules = cls._get_modules(name=name, group=group)

        snapshots = [
            snapshot for snapshot in cls._snapshot_names
            if len(filter(lambda x: x._has_snapshot(snapshot), modules)) == len(modules)
        ]

        return snapshots

    @classmethod
    def save_inputs(cls, name=None, group=None, snap_name="snapshot_0", saving=True,
                         max_saves=1000, bin_width=None, mode=None, label=None):
        """Saves inputs of registered modules."""
        modules = cls._get_modules(name=name, group=group)

        if not isinstance(label, dict):
            label = {module.name: label for module in modules}

        for module in modules:
            module.save_inputs(
                name=snap_name,
                saving=saving,
                max_saves=-1 if max_saves is None else max_saves,
                bin_width=bin_width,
                mode=mode,
                label=label[module.name],
            )

    @classmethod
    def save_gradients(cls, name=None, group=None, snap_name="snapshot_0", saving=True,
                           max_saves=1000, bin_width=None, input_label=None, output_label=None):
        """Saves gradients of registered modules."""

        modules = cls._get_modules(name=name, group=group)

        if not isinstance(input_label, dict):
            input_label = {module.name: input_label for module in modules}
        if not isinstance(output_label, dict):
            output_label = {module.name: output_label for module in modules}

        for module in modules:
            module.save_gradients(
                name=snap_name,
                saving=saving,
                max_saves=-1 if max_saves is None else max_saves,
                bin_width=bin_width,
                label_in=input_label[module.name],
                label_out=output_label[module.name],
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
    def create_snapshot(cls, name=None, group=None, snap_name="snapshot_0",
                        other_func=None, max_saves=1000, bin_width=None, label_in=None,
                        label_grad_in=None, label_grad_out=None, irm="layer",
                        function=False, inputs=False, gradients=False):
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
                label=label_in,
            )
        if gradients:
            cls.save_gradients(
                name=name,
                group=group,
                snap_name=snap_name,
                saving=True,
                max_saves=max_saves,
                bin_width=bin_width,
                input_label=label_grad_in,
                output_label=label_grad_out,
            )

    @classmethod
    def stop_saving(cls, name=None, group=None, inputs=True, gradients=True):
        """Stop retrieving inputs/gradients.
        
        Removes all forward/backward handles attached to modules.
        
        Args:
            name (str or list(str), optional):
            group (str or list(str), optional):
            inputs (bool): If ``True``, stop retrieving inputs.
            gradients (bool): If ``True``, stop retrieving gradients.
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
                      step=None, x_label=None, y_label=None, ax_title=False, function=False,
                      inputs=False, gradients_input=False, gradients_output=False, x_mode="expand",
                      tol_in=0.001, tol_grad_in=0.001, tol_grad_out=0.001, save_to=None, use_kde=False, **fig_kw):
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
            layout ("auto" or tuple(n_rows, n_cols)): Determines the `subplot layout`_. If ``layout=="auto"``
                the number of rows and columns is determined automatically. Otherwise ``n_rows*n_cols``
                should be greater than number of modules to plot or equal to 1, in which case all modules
                will be plotted in single plot.
            writer ():
            step ():
            x_label, y_label (str or dict(str, str)):
            ax_title (bool):
            x_mode (str): Determines how x axis range is determined between given :param:``x`` and range of input.
                Can be one of ``'expand', 'clip', 'x'``:
                * ``'expand'``: Always expand to greatest range.
                * ``'clip``: Always clip to smallest range.
                * ``'x'``: Use only :param:``x``.
                Defaults to ``"expand"``.
            tol_in, tol_grad_in, tol_grad_out (float):
            save_to (str): 
            use_kde (bool):
            fig_kw: Keyword arguments passed to :func:``matplotlib.pyplot.plt.subplots``. If ``axes is not None`` ignored.


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
                )

            if gradients_input:
                module.show_input_gradients(
                    axis=twin_x,
                    name=snap_name[module.name],
                    tolerance=tol_grad_in,
                    use_kde=use_kde,
                )
            if gradients_output:
                module.show_output_gradients(
                    axis=twin_x,
                    name=snap_name[module.name],
                    tolerance=tol_grad_out,
                    use_kde=use_kde,
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
                )

            if other_func is not None:
                for other_func_name in other_func:
                    axis.plot(
                        x_,
                        other_func[other_func_name](x_),
                        label=other_func_name,
                    )
            
            axis.set_xlim((min_x, max_x))

            if ax_title:
                axis.set_title(snap_name)
            if x_label[module.name] is not None:
                axis.set_xlabel(x_label)
            if y_label[module.name] is not None:
                axis.set_ylabel(y_label)

        if fig is not None:
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
                                function=True, inputs=False, gradients_input=False, gradients_output=False, **kwargs):
        """Creates an animation of plots over multiple `snapshots`.
        
        Args:
            path (str or pathlike): File to save animation to.
            snap_names (list(str) or list(dict(str, str))): All snapshots to animate over.
                Each element should be accepted by :func:``~activation_module.ActivationModule.show_function``.
            layout (str or tuple):
            video_writer ():
            step ():
            tag (str):
            kwargs: Keyword arguments passed to :func:``~activation_module.ActivationModule.show_function``.
                * `Kwarg` ``axes`` is ignored.
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