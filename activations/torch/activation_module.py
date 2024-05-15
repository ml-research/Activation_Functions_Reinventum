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


_LINED = dict()


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

    def show(self, x=None, fitted_function=True, other_func=None,
             title=None, axis=None, label=None, color=None):
        #Construct x axis
        if x is None:
            x = torch.arange(-3., 3, 0.01)
        elif isinstance(x, tuple) and len(x) in (2, 3):
            x = torch.arange(*x).float()
        elif isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
            x = torch.tensor(x.float())
        if axis is None:
            with sns.axes_style("whitegrid"):
                # fig, axis = plt.subplots(1, 1, figsize=(8, 6))
                fig, axis = plt.subplots(1, 1, figsize=(20, 12))
        if self.distributions:
            if self.distribution_display_mode in ["kde", "bar"]:
                ax2 = axis.twinx()
                if "neurons" in self.input_retrieval_mode:
                    x = self.plot_layer_distributions(ax2)
                else:
                    x = self.plot_distributions(ax2, color)
            elif self.distribution_display_mode == "points":
                x0, x_last, _ = self.get_distributions_range()
                x_edges = torch.tensor([x0, x_last]).float()
                y_edges = self.forward(x_edges.to(self.device)).detach().cpu().numpy()
                axis.scatter(x_edges, y_edges, color=color)
        #TODO: this should enable showing without input data from before
        y = self.forward(x.to(self.device)).detach().cpu().numpy()
        axis.plot(x, y, label=label, color=color)
        
        if writer is not None:
            try:
                writer.add_figure(title, fig, step)
            except AttributeError:
                self.logger.error("Could not use the given SummaryWriter to add the Rational figure")
        elif display:
            plt.show()
        else:
            if axis is None:
                return fig
            
    def plot_distributions(self, ax, colors=None, bin_size=None):
        """
        Plot the distribution and returns the corresponding x
        """
        ax.set_yticks([])
        try:
            import scipy.stats as sts
            scipy_imported = True
        except ImportError:
            RationalImportScipyWarning.warn()
            scipy_imported = False
        dists_fb = []
        x_min, x_max = np.inf, -np.inf
        #TODO: this is obsolete afaik
        """ if colors is None:
            colors = self.histograms_colors """
        if not(isinstance(colors, list) or isinstance(colors, tuple)):
            colors = create_colors(len(self.distributions))
        for i, (distribution, inp_label, color) in enumerate(zip(self.distributions, self.categories, colors)):
            if distribution.is_empty:
                if self.distribution_display_mode == "kde" and scipy_imported:
                    fill = ax.fill_between([], [], label=inp_label,  alpha=0.)
                else:
                    fill = ax.bar([], [], label=inp_label,  alpha=0.)
                dists_fb.append(fill)
            else:
                weights, x = _cleared_arrays(distribution.weights, distribution.bins, 0.001)
                # weights, x = distribution.weights, distribution.bins
                if self.distribution_display_mode == "kde" and scipy_imported:
                    if len(x) > 5:
                        refined_bins = np.linspace(x[0], x[-1], 200)
                        kde_curv = distribution.kde()(refined_bins)
                        # ax.plot(refined_bins, kde_curv, lw=0.1)
                        fill = ax.fill_between(refined_bins, kde_curv, alpha=0.45,
                                               color=color, label=inp_label)
                    else:
                        self.logger.warn(f"The bin size is too big, bins contain too few "
                              "elements.\nbins: {x}")
                        fill = ax.bar([], []) # in case of remove needed
                    size = x[1] - x[0]
                else:
                    width = (x[1] - x[0])/len(self.distributions)
                    if len(x) == len(weights):
                        fill = ax.bar(x+i*width, weights/weights.max(), width=width,
                                  linewidth=0, alpha=0.7, label=inp_label)
                    else:
                        fill = ax.bar(x[1:]+i*width, weights/weights.max(), width=width,
                                  linewidth=0, alpha=0.7, label=inp_label)
                    size = (x[1] - x[0])/100 # bar size can be larger
                dists_fb.append(fill)
                x_min, x_max = min(x_min, x[0]), max(x_max, x[-1])
        if self.distribution_display_mode in ["kde", "bar"]:
            leg = ax.legend(fancybox=True, shadow=True)
            leg.get_frame().set_alpha(0.4)
            for legline, origline in zip(leg.get_patches(), dists_fb):
                legline.set_picker(5)  # 5 pts tolerance
                _LINED[legline] = origline
            fig = plt.gcf()
            def toggle_distribution(event):
                # on the pick event, find the orig line corresponding to the
                # legend proxy line, and toggle the visibility
                leg = event.artist
                orig = _LINED[leg]
                if "get_visible" in dir(orig):
                    vis = not orig.get_visible()
                    orig.set_visible(vis)
                    color = orig.get_facecolors()[0]
                else:
                    vis = not orig.patches[0].get_visible()
                    color = orig.patches[0].get_facecolor()
                    for p in orig.patches:
                        p.set_visible(vis)
                # Change the alpha on the line in the legend so we can see what lines
                # have been toggled
                if vis:
                    leg.set_alpha(0.4)
                else:
                    leg.set_alpha(0.)
                leg.set_facecolor(color)
                fig.canvas.draw()
            fig.canvas.mpl_connect('pick_event', toggle_distribution)
        if x_min == np.inf or x_max == np.inf:
            torch.arange(-3, 3, 0.01)
        #TODO: when distribution is always empty, size wont be assigned and will throw an error

        return torch.arange(x_min, x_max, size)

    def plot_layer_distributions(self, ax):
        """
        Plot the layer distributions and returns the corresponding x
        """
        ax.set_yticks([])
        try:
            import scipy.stats as sts
            scipy_imported = True
        except ImportError:
            RationalImportScipyWarning.warn()
        dists_fb = []
        for distribution, inp_label, color in zip(self.distributions, self.categories, self.histograms_colors):
            #TODO: why is there no empty distribution check here?
            for n, (weights, x) in enumerate(zip(distribution.weights, distribution.bins)):
                if self.use_kde and scipy_imported:
                    if len(x) > 5:
                        refined_bins = np.linspace(float(x[0]), float(x[-1]), 200)
                        kde_curv = distribution.kde(n)(refined_bins)
                        # ax.plot(refined_bins, kde_curv, lw=0.1)
                        fill = ax.fill_between(refined_bins, kde_curv, alpha=0.4,
                                                color=color, label=f"{inp_label} ({n})")
                    else:
                        self.logger.warn(f"The bin size is too big, bins contain too few "
                              "elements.\nbins: {x}")
                        fill = ax.bar([], []) # in case of remove needed
                else:
                    fill = ax.bar(x, weights/weights.max(), width=x[1] - x[0],
                                  linewidth=0, alpha=0.4, color=color,
                                  label=f"{inp_label} ({n})")
                dists_fb.append(fill)

        if self.distribution_display_mode in ["kde", "bar"]:
            leg = ax.legend(fancybox=True, shadow=True)
            leg.get_frame().set_alpha(0.4)
            for legline, origline in zip(leg.get_patches(), dists_fb):
                legline.set_picker(5)  # 5 pts tolerance
                _LINED[legline] = origline
            fig = plt.gcf()
            def toggle_distribution(event):
                # on the pick event, find the orig line corresponding to the
                # legend proxy line, and toggle the visibility
                leg = event.artist
                orig = _LINED[leg]
                if "get_visible" in dir(orig):
                    vis = not orig.get_visible()
                    orig.set_visible(vis)
                    color = orig.get_facecolors()[0]
                else:
                    vis = not orig.patches[0].get_visible()
                    color = orig.patches[0].get_facecolor()
                    for p in orig.patches:
                        p.set_visible(vis)
                # Change the alpha on the line in the legend so we can see what lines
                # have been toggled
                if vis:
                    leg.set_alpha(0.4)
                else:
                    leg.set_alpha(0.)
                leg.set_facecolor(color)
                fig.canvas.draw()
            fig.canvas.mpl_connect('pick_event', toggle_distribution)
        return torch.arange(*self.get_distributions_range())
    
    def show_gradients(self, display=True, tolerance=0.001, title=None,
                       axis=None, writer=None, step=None, label=None, colors=None):
        try:
            import scipy.stats as sts
            scipy_imported = True
        except ImportError:
            RationalImportScipyWarning.warn()
            scipy_imported = False
        if axis is None:
            with sns.axes_style("whitegrid"):
                # fig, axis = plt.subplots(1, 1, figsize=(8, 6))
                fig, axis = plt.subplots(1, 1, figsize=(20, 12))
        if colors is None or len(colors) != 2:
            colors = ["orange", "blue"]
        dists = [self._in_grad_dist, self._out_grad_dist]
        if label is None:
            labels = ['input grads', 'output grads']
        else:
            labels = [f'{label} (inp)', f'{label} (outp)']
        for distribution, col, label in zip(dists, colors, labels):
            weights, x = distribution.weights, distribution.bins
            if self.use_kde and scipy_imported:
                if len(x) > 5:
                    refined_bins = np.linspace(float(x[0]), float(x[-1]), 200)
                    kde_curv = distribution.kde()(refined_bins)
                    # ax.plot(refined_bins, kde_curv, lw=0.1)
                    axis.fill_between(refined_bins, kde_curv, alpha=0.4,
                                      color=col, label=label)
                else:
                    self.logger.warn("The bin size is too big, bins contain too few "
                                     f"elements.\nbins: {x}")
                    axis.bar([], []) # in case of remove needed
            else:
                axis.bar(x, weights/weights.max(), width=x[1] - x[0],
                         linewidth=0, alpha=0.4, color=col, label=label)
            distribution.empty()
        if writer is not None:
            try:
                writer.add_figure(title, fig, step)
            except AttributeError:
                self.logger.error("Could not use the given SummaryWriter to add the Rational figure")
        elif display:
            plt.legend()
            plt.show()
        else:
            if axis is None:
                return fig
            
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

    def export_graph(self, path="rational_function.svg", snap_number=-1,
                     other_func=None):
        """
        Saves one graph of the function based on the last snapshot \
        (by default, and if available).

        Arguments:
                path (str):
                    Complete path with name of the figure.\n
                    Default ``"rational_functions.svg"``
                together (bool):
                    If True, the graphs of every functions are stored in \
                    different files.\n
                    Default ``True``
                layout (tuple or 'auto'):
                    Grid layout of the figure. If "auto", one is generated.\
                    (see `layout`).
                    Default ``auto``
                snap_number (int):
                    The snap to take in snapshot_list for each function.\n
                    Default ``-1 (last)``
                other_func (callable):
                    another function to be plotted or a list of other callable
                    functions or a dictionary with the function name as key
                    and the callable as value.
                    Default ``None``
        """
        if not len(self.snapshots):
            mes =("Cannot use the last snapshot as the snapshot_list "
                  "is empty, making a capture with default params")
            RationalWarning.warn(mes)
            self.capture()
        snap = self.snapshots[snap_number]
        snap.save(path=path, other_func=other_func)

    def export_evolution_graph(self, path="rational_evolution.gif",
                               animated=True, other_func=None):
        """
        Creates and saves an animated graph of the function evolution based \
        on the successive snapshots saved in `snapshot_list`.

        Arguments:
                path (str):
                    Complete path with name of the figure.\n
                    Default ``"rational_evolution.gif"``
                animated (bool):
                    Complete path with name of the figure.\n
                    Default ``True``
                other_func (callable):
                    another function to be plotted or a list of other callable
                    functions or a dictionary with the function name as key
                    and the callable as value. \n
                    Default ``None``
        """
        if animated:
            import io
            from PIL import Image
            if len(self.snapshots) < 2:
                print("Cannot save a gif as you have taken less than 1 snapshot")
                return
            fig = plt.gcf()
            x_min, x_max, y_min, y_max = _get_frontiers(self.snapshots,
                                                        other_func)
            input = np.arange(x_min, x_max, (x_max - x_min)/10000)
            gif_images = []
            for i, snap in enumerate(self.snapshots):
                fig = snap.show(x=input, other_func=other_func, display=False,
                                duplicate_axis=self.use_multiple_axis)
                ax0 = fig.axes[0]
                ax0.set_xlim([x_min, x_max])
                ax0.set_ylim([y_min, y_max])
                buf = io.BytesIO()
                fig.set_tight_layout(True)
                plt.savefig(buf, format='png')
                buf.seek(0)
                gif_images.append(Image.open(buf))
                fig.clf()
            if path[-4:] != ".gif":
                path += ".gif"
            path = _repair_path(path)
            gif_images[0].save(path, save_all=True, duration=800, loop=0,
                               append_images=gif_images[1:], optimize=False)
        else:
            if path[-4:] == ".gif":
                path = path[-4:] + ".svg"
            path = _path_for_multiple(path, "evolution")
            for i, snap in enumerate(self.snapshots):
                pos = path.rfind(".")
                if pos > 0:
                    new_path = f"{path[pos:]}_{i}{path[:pos]}"
                else:
                    new_path = f"{path}_{i}"
                snap.save(path=new_path, other_func=other_func)

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
    def show_all(cls, x=None, fitted_function=True, other_func=None,
                 display=True, tolerance=0.001, title=None, axes=None,
                 layout="auto", writer=None, step=None, colors="#1f77b4"):
        """
        Shows a graph of the all instanciated activation functions (or returns \
        it if ``returns=True``).

        Arguments:
                x (range):
                    The range to print the function on.\n
                    Default ``None``
                fitted_function (bool):
                    If ``True``, displays the best fitted function if searched.
                    Otherwise, returns it. \n
                    Default ``True``
                other_funcs (callable):
                    another function to be plotted or a list of other callable
                    functions or a dictionary with the function name as key
                    and the callable as value.
                display (bool):
                    If ``True``, displays the plot.
                    Otherwise, returns the figure. \n
                    Default ``False``
                tolerance (float):
                    If the input histogram is used, it will be pruned. \n
                    Every bin containg less than `tolerance` of the total \
                    input is pruned out.
                    (Reduces noise).
                    Default ``0.001``
                title (str):
                    If not None, a title for the figure
                    Default ``None``
                axes (matplotlib.pyplot.axis):
                    On ax or a list of axes to be plotted on. \n
                    If None, creates them automatically (see `layout`). \n
                    Default ``None``
                layout (tuple or 'auto'):
                    Grid layout of the figure. If "auto", one is generated.\n
                    Default ``"auto"``
                writer (tensorboardX.SummaryWriter):
                    A tensorboardX writer to give the image to, in case of
                    debugging.
                    Default ``None``
                step (int):
                    A step/epoch for tensorboardX writer.
                    If None, incrementing itself.
                    Default ``None``
        """
        logger = ActivationLogger(f"{cls.__name__}Logger")
        instances_list = cls._get_instances()
        if axes is None:
            if layout == "auto":
                total = len(instances_list)
                layout = _get_auto_axis_layout(total)
            if len(layout) != 2:
                msg = 'layout should be either "auto" or a tuple of size 2'
                raise TypeError(msg)
            figs = tuple(np.flip(np.array(layout)* (2, 3)))
            try:
                import seaborn as sns
                with sns.axes_style("whitegrid"):
                    fig, axes = plt.subplots(*layout, figsize=figs)
            except ImportError:
                logger.warn("Could not import seaborn")
                #RationalImportSeabornWarning.warn()
                fig, axes = plt.subplots(*layout, figsize=figs)
            if isinstance(axes, plt.Axes):
                axes = np.array([axes])
            # if display:
            for ax in axes.flatten()[len(instances_list):]:
                ax.remove()
            axes = axes[:len(instances_list)]
        elif isinstance(axes, plt.Axes):
            axes = np.array([axes for _ in range(len(instances_list))])
            fig = plt.gcf()
        if isinstance(colors, str):
            colors = [colors]*len(axes.flatten())
        if isinstance(x, list):
            for act, ax, x_act, color in zip(instances_list, axes.flatten(), x, colors):
                act.show(x_act, fitted_function, other_func, False, tolerance,
                         title, axis=ax, writer=None, step=step,
                         color=color)
        else:
            for act, ax, color in zip(instances_list, axes.flatten(), colors):
                act.show(x, fitted_function, other_func, False, tolerance,
                         title, axis=ax, writer=None, step=step,
                         color=color)
        if title is not None:
            fig.suptitle(title, y=0.95)
        fig = plt.gcf()
        fig.tight_layout()
        if writer is not None:
            if step is None:
                step = cls._step
                cls._step += 1
            writer.add_figure(title, fig, step)
        elif display:
            # plt.legend()
            plt.show()
        else:
            return fig

    @classmethod
    def show_all_gradients(cls, display=True, tolerance=0.001, title=None,
                           axes=None, layout="auto", writer=None, step=None,
                           colors=None):
        """
        Shows a graph of the all instanciated activation functions (or returns \
        it if ``returns=True``).

        Arguments:
                x (range):
                    The range to print the function on.\n
                    Default ``None``
                fitted_function (bool):
                    If ``True``, displays the best fitted function if searched.
                    Otherwise, returns it. \n
                    Default ``True``
                other_funcs (callable):
                    another function to be plotted or a list of other callable
                    functions or a dictionary with the function name as key
                    and the callable as value.
                display (bool):
                    If ``True``, displays the plot.
                    Otherwise, returns the figure. \n
                    Default ``False``
                tolerance (float):
                    If the input histogram is used, it will be pruned. \n
                    Every bin containg less than `tolerance` of the total \
                    input is pruned out.
                    (Reduces noise).
                    Default ``0.001``
                title (str):
                    If not None, a title for the figure
                    Default ``None``
                axes (matplotlib.pyplot.axis):
                    On ax or a list of axes to be plotted on. \n
                    If None, creates them automatically (see `layout`). \n
                    Default ``None``
                layout (tuple or 'auto'):
                    Grid layout of the figure. If "auto", one is generated.\n
                    Default ``"auto"``
                writer (tensorboardX.SummaryWriter):
                    A tensorboardX writer to give the image to, in case of
                    debugging.
                    Default ``None``
                step (int):
                    A step/epoch for tensorboardX writer.
                    If None, incrementing itself.
                    Default ``None``
        """
        logger = ActivationLogger("f{cls.__name__}Logger")
        instances_list = cls._get_instances()
        if axes is None:
            if layout == "auto":
                total = len(instances_list)
                layout = _get_auto_axis_layout(total)
            if len(layout) != 2:
                msg = 'layout should be either "auto" or a tuple of size 2'
                raise TypeError(msg)
            figs = tuple(np.flip(np.array(layout)* (2, 3)))
            try:
                import seaborn as sns
                with sns.axes_style("whitegrid"):
                    fig, axes = plt.subplots(*layout, figsize=figs)
            except ImportError:
                logger.warn("Could not import seaborn")
                #RationalImportSeabornWarning.warn()
                fig, axes = plt.subplots(*layout, figsize=figs)
            if isinstance(axes, plt.Axes):
                axes = np.array([axes])
            # if display:
            for ax in axes.flatten()[len(instances_list):]:
                ax.remove()
            axes = axes[:len(instances_list)]
        elif isinstance(axes, plt.Axes):
            axes = np.array([axes for _ in range(len(instances_list))])
            fig = plt.gcf()
        if isinstance(colors, str) or colors is None:
            colors = [colors]*len(axes.flatten())
        for act, ax, color in zip(instances_list, axes.flatten(), colors):
            act.show_gradients(False, tolerance, title, axis=ax,
                               writer=None, step=step, colors=color)
        if title is not None:
            fig.suptitle(title, y=0.95)
        fig = plt.gcf()
        fig.tight_layout()
        if writer is not None:
            if step is None:
                step = cls._step
                cls._step += 1
            writer.add_figure(title, fig, step)
        elif display:
            plt.legend()
            plt.show()
        else:
            return fig
        
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

    @classmethod
    def export_graphs(cls, path="rational_functions.svg", together=True,
                      layout="auto", snap_number=-1, other_func=None):
        """
        Saves one or more graph(s) of the function based on the last snapshot \
        (by default, and if available) for each instanciated rational function.

        Arguments:
                path (str):
                    Complete path with name of the figure.\n
                    Default ``"rational_functions.svg"``
                together (bool):
                    If True, the graphs of every functions are stored in \
                    different files.\n
                    Default ``True``
                layout (tuple or 'auto'):
                    Grid layout of the figure. If "auto", one is generated.\
                    (see `layout`).
                    Default ``"auto"``
                snap_number (int):
                    The snap to take in snapshot_list for each function.\n
                    Default ``-1 (last)``
                other_func (callable):
                    another function to be plotted or a list of other callable
                    functions or a dictionary with the function name as key
                    and the callable as value.
                    Default ``None``
        """
        if together:
            for i, rat in enumerate(cls.list):
                if not len(rat.snapshot_list) > 0:
                    print(f"Cannot use the last snapshots as snapshot n {i} \
                          is empty, capturing...")
                    cls.capture_all()
                    break
            if layout == "auto":
                total = len(cls.list)
                layout = _get_auto_axis_layout(total)
            if len(layout) != 2:
                msg = 'layout should be either "auto" or a tuple of size 2'
                raise TypeError(msg)
            figs = tuple(np.flip(np.array(layout) * (2, 3)))
            try:
                import seaborn as sns
                with sns.axes_style("whitegrid"):
                    fig, axes = plt.subplots(*layout, figsize=figs)
            except ImportError:
                RationalImportSeabornWarning.warn()
                fig, axes = plt.subplots(*layout, figsize=figs)
            for rat, ax in zip(cls.list, axes.flatten()):
                snap = rat.snapshot_list[snap_number]
                snap.show(display=False, axis=ax, other_func=other_func,
                          duplicate_axis=cls.use_multiple_axis)
            for ax in axes.flatten()[len(cls.list):]:
                ax.remove()
            fig.savefig(_repair_path(path))
            fig.clf()
        else:
            path = _path_for_multiple(path, "graphs")
            for i, rat in enumerate(tqdm(cls.list, desc="Saving Rationals")):
                pos = path.rfind(".")
                new_path = f"{path[:pos]}_{i}{path[pos:]}"
                rat.export_graph(new_path)

    @classmethod
    def export_evolution_graphs(cls, path="rationals_evolution.gif",
                                together=True, layout="auto", animated=True,
                                other_func=None):
        """
        Creates and saves an animated graph of the function evolution based \
        on the successive snapshots saved in `snapshot_list` for each \
        instanciated rational function.

        Arguments:
                path (str):
                    Complete path with name of the figure.\n
                    Default ``"rationals_evolution.gif"``
                together (bool):
                    If True, the graphs of every functions are stored in \
                    different files.\n
                    Default ``True``
                layout (tuple or 'auto'):
                    Grid layout of the figure. If "auto", one is generated.\
                    (see `layout`).\n
                    Default ``"auto"``
                animated (bool):
                    If True, creates an animated gif, else, different files \
                    are created.\n
                    Default ``True``
                other_func (callable):
                    another function to be plotted or a list of other \
                    callable functions or a dictionary with the function \
                    name as key and the callable as value.\n
                    Default ``None``
        """
        if animated:
            if together:
                nb_sn = len(cls.list[0].snapshot_list)
                if any([len(rat.snapshot_list) != nb_sn for rat in cls.list]):
                    msg = "Seems that not all rationals have the same " \
                          "number of snapshots."
                    RationalWarning.warn(msg)
                import io
                from PIL import Image
                limits = []
                for i, rat in enumerate(cls.list):
                    if len(rat.snapshot_list) < 2:
                        msg = "Cannot save a gif as you have taken less " \
                              f"than 1 snapshot for rational n {i}"
                        print(msg)
                        return
                    limits.append(_get_frontiers(rat.snapshot_list,
                                                 other_func))
                if layout == "auto":
                    total = len(cls.list)
                    layout = _get_auto_axis_layout(total)
                if len(layout) != 2:
                    msg = 'layout should be either "auto" or a tuple of size 2'
                    raise TypeError(msg)
                fig = plt.gcf()
                gif_images = []
                seaborn_installed = True
                try:
                    import seaborn as sns
                except ImportError:
                    seaborn_installed = False
                    RationalImportSeabornWarning.warn()
                if seaborn_installed:
                    with sns.axes_style("whitegrid"):
                        figs = tuple(np.flip(np.array(layout)* (2, 3)))
                        fig, axes = plt.subplots(*layout, figsize=figs)
                else:
                    figs = tuple(np.flip(np.array(layout)* (2, 3)))
                    fig, axes = plt.subplots(*layout, figsize=figs)
                for ax in axes.flatten()[len(cls.list):]:
                    ax.remove()  # removes empty axes
                for i in range(nb_sn):
                    for rat, ax, lim in zip(cls.list, axes.flatten(), limits):
                        x_min, x_max, y_min, y_max = lim
                        input = np.arange(x_min, x_max, (x_max - x_min)/10000)
                        snap = rat.snapshot_list[i]
                        snap.show(x=input, other_func=other_func,
                                  display=False, axis=ax,
                                  duplicate_axis=cls.use_multiple_axis)
                        ax.set_xlim([x_min, x_max])
                        ax.set_ylim([y_min, y_max])
                    buf = io.BytesIO()
                    fig.set_tight_layout(True)
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    gif_images.append(Image.open(buf))
                    for i, ax in enumerate(fig.axes):
                        if i < len(cls.list):
                            ax.cla()
                        else:
                            ax.remove()
                if path[-4:] != ".gif":
                    path += ".gif"
                path = _repair_path(path)
                gif_images[0].save(path, save_all=True, duration=800, loop=0,
                                   append_images=gif_images[1:], optimize=False)
            else:
                path = _path_for_multiple(path, "graphs")
                bar_title = "Saving Rationals' evolutions"
                for i, rat in enumerate(tqdm(cls.list, desc=bar_title)):
                    pos = path.rfind(".")
                    if pos > 0:
                        new_path = f"{path[:pos]}_{i}{path[pos:]}"
                    else:
                        new_path = f"{path}_{i}"
                    rat.export_evolution_graph(new_path, True, other_func)
        else:  # not animated
            if path[-4:] == ".gif":
                path = path[-4:] + ".svg"
            path = _path_for_multiple(path, "evolution")
            if together:
                nb_sn = len(cls.list[0].snapshot_list)
                if any([len(rat.snapshot_list) != nb_sn for rat in cls.list]):
                    msg = "Seems that not all rationals have the " \
                          "same number of snapshots."
                    RationalWarning.warn(msg)
                for snap_number in range(nb_sn):
                    if "." in path:
                        ext = path.split(".")[-1]
                        main = ".".join(path.split(".")[:-1])
                        new_path = f"{main}_{snap_number}.{ext}"
                    else:
                        new_path = f"{path}_{snap_number}"
                    cls.export_graphs(new_path, together, layout, snap_number,
                                      other_func)
            else:
                for i, rat in enumerate(cls.list):
                    pos = path.rfind(".")
                    if pos > 0:
                        new_path = f"{path[pos:]}_{i}{path[:pos]}"
                    else:
                        new_path = f"{path}_{i}"
                    rat.export_evolution_graph(new_path, False, other_func)

    def state_dict(self, destination=None, *args, **kwargs):
        _state_dict = super().state_dict(destination, *args, **kwargs)
        if self.distributions is not None:
            _state_dict["distributions"] = self.distributions
        return _state_dict



if __name__ == '__main__':
    def plot_gaussian(mode, device):
        _2pi_sqrt = 2.5066
        tanh = torch.tanh
        relu = F.relu

        nb_neurons_in_layer = 5

        leaky_relu = F.leaky_relu
        gaussian = lambda x: torch.exp(-0.5*x**2) / _2pi_sqrt
        gaussian.__name__ = "gaussian"
        gau = ActivationModule(gaussian, device=device)
        gau.save_inputs(mode=mode, category_name="neg") # Wrong
        inp = torch.stack([(torch.rand(10000)-(i+1))*2 for i in range(nb_neurons_in_layer)], 1)
        print(inp.shape)
        gau(inp.to(device))
        if "categories" in mode:
            gau.current_inp_category = "pos"
            inp = torch.stack([(torch.rand(10000)+(i+1))*2 for i in range(nb_neurons_in_layer)], 1)
            gau(inp.to(device))
            # gau(inp.cuda())
        gau.show()

    ActivationModule.distribution_display_mode = "bar"
    # for device in ["cuda:0", "cpu"]:
    for device in ["cpu"]:
        for mode in ["categories", "neurons", "neurons_categories"]:
            plot_gaussian(mode, device)
