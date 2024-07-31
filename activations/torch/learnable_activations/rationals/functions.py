import torch



def rational_A(x, weight_numerator, weight_denominator):
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    x_powers = torch.pow(x.view(-1, 1), torch.arange(max(len_num, len_deno + 1), device=x.device))
    numerator = torch.mul(x_powers[:, :len_num], weight_numerator).sum(1)
    denominator = torch.mul(x_powers[:, 1:len_deno+1], weight_denominator).abs().sum(1)
    return torch.div(numerator, denominator + 1.0).view(x.shape)


def rational_B(x, weight_numerator, weight_denominator):
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    x_powers = torch.pow(x.view(-1, 1), torch.arange(max(len_num, len_deno + 1), device=x.device))
    numerator = torch.mul(x_powers[:, :len_num], weight_numerator).sum(1)
    denominator = torch.mul(x_powers[:, 1:len_deno+1], weight_denominator).sum(1).abs()
    return torch.div(numerator, denominator + 1.0).view(x.shape)


def rational_C(x, weight_numerator, weight_denominator):
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    x_powers = torch.pow(x.view(-1, 1), torch.arange(max(len_num, len_deno), device=x.device))
    numerator = torch.mul(x_powers[:, :len_num], weight_numerator).sum(1)
    denominator = torch.mul(x_powers[:, :len_deno], weight_denominator).sum(1).abs()
    return torch.div(numerator, denominator).view(x.shape)


def rational_D(x, weight_numerator, weight_denominator, *, noise_deviation=0.2):
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    x_powers = torch.pow(x.view(-1, 1), torch.arange(max(len_num, len_deno), device=x.device))

    noise = torch.FloatTensor(len_num).uniform_(1-noise_deviation, 1+noise_deviation)
    noised_numerator = torch.mul(weight_numerator, noise)

    numerator = torch.mul(x_powers[:, :len_num], noised_numerator).sum(1)
    denominator = torch.mul(x_powers[:, 1:len_deno+1], weight_denominator).sum(1).abs()
    return torch.div(numerator, denominator + 1.0).view(x.shape)


def rational_nonsafe(x, weight_numerator, weight_denominator):
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    x_powers = torch.pow(x.view(-1, 1), torch.arange(max(len_num, len_deno + 1), device=x.device))
    numerator = torch.mul(x_powers[:, :len_num], weight_numerator).sum(1)
    denominator = torch.mul(x_powers[:, 1:len_deno+1], weight_denominator).sum(1)
    return torch.div(numerator, denominator + 1.0).view(x.shape)


def rational_spline(x, weight_numerator, weight_denominator, *, k=2.0):
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    x_powers = torch.pow(x.view(-1, 1), torch.arange(max(len_num, len_deno + 1), device=x.device))
    z = x.view(-1)
    numerator = torch.mul(x_powers[:, :len_num], weight_numerator).sum(1).mul(torch.relu(z+k)).mul(-torch.relu(-z+k))
    denominator = torch.mul(x_powers[:, 1:len_deno+1], weight_denominator).sum(1).abs()
    return torch.div(numerator, denominator + 1.0).view(x.shape)
