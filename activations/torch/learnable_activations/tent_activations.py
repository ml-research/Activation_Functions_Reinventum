import torch
import torch.nn as nn



def tent_activation(x, delta):
    """
    Functional implementation of TentActivation.
    """
    return torch.clamp(delta - torch.abs(x), min=0)


class TentActivation(nn.Module):
    def __init__(self, delta=2.0, learnable=False):
        super().__init__()

        if torch.is_tensor(delta):
            self.delta = nn.Parameter(delta, requires_grad=learnable)
        else:
            self.delta = nn.Parameter(torch.tensor(delta), requires_grad=learnable)

    @property
    def learnable(self):
        return self.delta.requires_grad

    def forward(self, x):
        return tent_activation(x, self.delta)


def bitent_activation(x, delta, epsilon):
    """
    Functional implementation of BiTentActivation.
    """
    hdt = delta/2
    return tent_activation(x+hdt+epsilon, hdt) + tent_activation(x-hdt-epsilon, hdt)


class BiTentActivation(TentActivation):
    def __init__(self, delta=2.0, learnable=False):
        super().__init__(delta, learnable)
        self.epsilon = 0.05

    def forward(self, x):
        return bitent_activation(x, self.delta, self.epsilon)
