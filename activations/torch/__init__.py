import os

from .activation_module import ActivationModule

ActivationModule.set_plotting_style(path = os.path.join(__file__, "..", "..", "..", "modified_whitegrid.mplstyle"))