import torch



def rational_A(x, weight_numerator, weight_denominator):
    """Computes :math:`f(x)=\\frac{\\sum_i^Na_ix^i}{1+\\sum_i^M|b_ix^i|}`.
    
    Args:
        x (:class:`torch.Tensor`):
            Inputs of any shape. Will be treated as 1D tensor.

        weight_numerator (:class:`torch.Tensor`):
            Tensor of shape :math:`N`.

        weight_denominator (:class:`torch.Tensor`):
            Tensor of shape :math:`M`.

    Returns:
        :class:`torch.Tensor`:
            Tensor of same shape as ``x``.
    """
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    x_powers = torch.pow(x.view(-1, 1), torch.arange(max(len_num, len_deno + 1), device=x.device))
    numerator = torch.mul(x_powers[:, :len_num], weight_numerator).sum(1)
    denominator = torch.mul(x_powers[:, 1:len_deno+1], weight_denominator).abs().sum(1)
    return torch.div(numerator, denominator + 1.0).view(x.shape)


def rational_B(x, weight_numerator, weight_denominator):
    """Computes :math:`f(x)=\\frac{\\sum_i^Na_ix^i}{1+|\\sum_i^Mb_ix^i|}`.
    
    Args:
        x (:class:`torch.Tensor`):
            Inputs of any shape. Will be treated as 1D tensor.

        weight_numerator (:class:`torch.Tensor`):
            Tensor of shape :math:`N`.

        weight_denominator (:class:`torch.Tensor`):
            Tensor of shape :math:`M`.

    Returns:
        :class:`torch.Tensor`:
            Tensor of same shape as ``x``.
    """
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    x_powers = torch.pow(x.view(-1, 1), torch.arange(max(len_num, len_deno + 1), device=x.device))
    numerator = torch.mul(x_powers[:, :len_num], weight_numerator).sum(1)
    denominator = torch.mul(x_powers[:, 1:len_deno+1], weight_denominator).sum(1).abs()
    return torch.div(numerator, denominator + 1.0).view(x.shape)


def rational_C(x, weight_numerator, weight_denominator):
    """Computes :math:`f(x)=\\frac{\\sum_i^Na_ix^i}{|\\sum_i^Mb_ix^i|}`.
    
    Args:
        x (:class:`torch.Tensor`):
            Inputs of any shape. Will be treated as 1D tensor.

        weight_numerator (:class:`torch.Tensor`):
            Tensor of shape :math:`N`.

        weight_denominator (:class:`torch.Tensor`):
            Tensor of shape :math:`M`.

    Returns:
        :class:`torch.Tensor`:
            Tensor of same shape as ``x``.
    """
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    x_powers = torch.pow(x.view(-1, 1), torch.arange(max(len_num, len_deno), device=x.device))
    numerator = torch.mul(x_powers[:, :len_num], weight_numerator).sum(1)
    denominator = torch.mul(x_powers[:, :len_deno], weight_denominator).sum(1).abs()
    return torch.div(numerator, denominator).view(x.shape)


def rational_D(x, weight_numerator, weight_denominator, *, noise_deviation=0.2):
    """Computes version **B** but with noise added to numerator.

    .. math::
        \\begin{cases}
            \\text{let }c\\text{be sampled from uniform distribution }\\mathcal{U} \\\\
            f(x)=\\frac{\\sum_i^Na_ix^ic_i}{1+|\\sum_i^Mb_ix^i|}
        \\end{cases}
    
    Args:
        x (:class:`torch.Tensor`):
            Inputs of any shape. Will be treated as 1D tensor.

        weight_numerator (:class:`torch.Tensor`):
            Tensor of shape :math:`N`.

        weight_denominator (:class:`torch.Tensor`):
            Tensor of shape :math:`M`.

        noise_deviation (float):
            Specifies noise distribution :math:`\\mathcal{U}=\\frac{1}{2\\sigma}`.

    Returns:
        :class:`torch.Tensor`:
            Tensor of same shape as ``x``.
    """
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    x_powers = torch.pow(x.view(-1, 1), torch.arange(max(len_num, len_deno), device=x.device))

    noise = torch.FloatTensor(len_num).uniform_(1.0-noise_deviation, 1.0+noise_deviation)
    noised_numerator = torch.mul(weight_numerator, noise)

    numerator = torch.mul(x_powers[:, :len_num], noised_numerator).sum(1)
    denominator = torch.mul(x_powers[:, 1:len_deno+1], weight_denominator).sum(1).abs()
    return torch.div(numerator, denominator + 1.0).view(x.shape)


def rational_nonsafe(x, weight_numerator, weight_denominator):
    """Computes :math:`f(x)=\\frac{\\sum_i^Na_ix^i}{1+\\sum_i^Mb_ix^i}`.
    
    Args:
        x (:class:`torch.Tensor`):
            Inputs of any shape. Will be treated as 1D tensor.

        weight_numerator (:class:`torch.Tensor`):
            Tensor of shape :math:`N`.

        weight_denominator (:class:`torch.Tensor`):
            Tensor of shape :math:`M`.

    Returns:
        :class:`torch.Tensor`:
            Tensor of same shape as ``x``.

    .. warning::
        Can result in division by zero.
    """
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    x_powers = torch.pow(x.view(-1, 1), torch.arange(max(len_num, len_deno + 1), device=x.device))
    numerator = torch.mul(x_powers[:, :len_num], weight_numerator).sum(1)
    denominator = torch.mul(x_powers[:, 1:len_deno+1], weight_denominator).sum(1)
    return torch.div(numerator, denominator + 1.0).view(x.shape)


def rational_spline(x, weight_numerator, weight_denominator, *, k=2.0):
    """Computes :math:`f(x)=\\frac{\\sum_i^Na_ix^i\\cdot\\text{max}\\{0, x_i+k\\}\\cdot\\text{min}\\{0, x_i-k\\}}{1+|\\sum_i^Mb_ix^i|}`.
    
    Args:
        x (:class:`torch.Tensor`):
            Inputs of any shape. Will be treated as 1D tensor.

        weight_numerator (:class:`torch.Tensor`):
            Tensor of shape :math:`N`.

        weight_denominator (:class:`torch.Tensor`):
            Tensor of shape :math:`M`.

        k (float):
            Parameter :math:`k`.

    Returns:
        :class:`torch.Tensor`:
            Tensor of same shape as ``x``.
    """
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    x_powers = torch.pow(x.view(-1, 1), torch.arange(max(len_num, len_deno + 1), device=x.device))
    z = x.view(-1)
    numerator = torch.mul(x_powers[:, :len_num], weight_numerator).sum(1).mul(torch.relu(z+k)).mul(-torch.relu(-z+k))
    denominator = torch.mul(x_powers[:, 1:len_deno+1], weight_denominator).sum(1).abs()
    return torch.div(numerator, denominator + 1.0).view(x.shape)
