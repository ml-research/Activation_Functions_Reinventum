import numpy as np


def find_closest_equivalent(rational_func, new_func, x):
    """
    Compute the parameters a, b, c, and d that minimizes distance between the
    rational function and the other function on the range `x`

    Arguments:
            rational_func (callable):
                The rational function to consider.\n
            new_func (callable):
                The function you want to fit to rational.\n
            x (array):
                The range on which the curves of the functions are fitted
                together.\n
                Default ``True``
    Returns:
        tuple: ((a, b, c, d), dist) with: \n
        a, b, c, d: the parameters to adjust the function \
            (vertical and horizontal scales and bias) \n
        dist: The final distance between the rational function and the \
        fitted one
    """
    initials = np.array([1.0, 0.0, 1.0, 0.0])  # a, b, c, d
    y = rational_func(x)
    from scipy.optimize import curve_fit

    def equivalent_func(x_array, a, b, c, d):
        return a * new_func(c * x_array + d) + b

    params = curve_fit(equivalent_func, x, y, initials)
    a, b, c, d = params[0]
    final_func_output = np.array(equivalent_func(x, a, b, c, d))
    final_distance = np.sqrt(((y - final_func_output) ** 2).sum())
    return (a, b, c, d), final_distance


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
