import numpy as np



def _get_auto_axis_layout(nb_plots):
    if nb_plots == 1:
        return 1, 1
    mid = int(np.sqrt(nb_plots))
    for i in range(mid, 1, -1):
        mod = nb_plots % i
        if mod == 0:
            return i, nb_plots // i
    if mid * (mid + 1) >= nb_plots:
        return mid, mid + 1
    return mid + 1, mid + 1