import torch
import torch.nn as nn

from activations.torch import ActivationModule



class PReLUWrapper(nn.PReLU):
    def __init__(self, num_parameters: int = 1, init: float = 0.25, device=None, dtype=None) -> None:
        super().__init__(num_parameters, init, device, dtype)

        self.register_buffer("i", torch.zeros(1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # to verify that snapshots are created correctly (in debugger).
        self.i = self.i + 1
        return super().forward(input)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(2, 3)
        self.act = PReLUWrapper()

    def forward(self, x):
        return self.act(self.linear(x))


if __name__ == "__main__":    
    model = Model()

    ActivationModule.register(model.act, "actfn", display_mode="bar", irm="layer")

    optim = torch.optim.SGD(params=model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    data = torch.rand(50, 2, requires_grad=True)
    labels = torch.zeros(50, 3, requires_grad=False)
    batch_size = 5
    batches = [
        (data[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])
        for i in range(len(data))
        ]

    ActivationModule.show_function(name="actfn", snap_name=None, display=True,
                                x=(-3, 3, 30), function=True, title="Initialization")


    # create snapshot before training
    # gradients/input will be collected wrt. to initialization state
    # i.e. gradients/inputs of first epoch will be plotted with function at initialization
    ActivationModule.create_snapshot(
        name="actfn", snap_name=f"epoch_0", max_saves=10,
        function=True, gradients=False, inputs=True 
    )
    for e in range(5):
        for b, (x, y) in enumerate(batches):
            pred = model(x)
            loss = criterion(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

        ActivationModule.create_snapshot(
            name="actfn", snap_name=f"epoch_{e+1}", max_saves=10,
            function=True, gradients=False, inputs=True
        )


    for i in range(6):
        ActivationModule.show_function(
            name="actfn", snap_name=f"epoch_{i}", display=False, save_to=rf"epoch_{i}.png",
            function=True, inputs=True, gradients_input=False, gradients_output=False, title=f"Epoch {i}"
        )

    ActivationModule.export_evolution_graphs(
        name="actfn", path=r"test.gif",
        snap_names=[f"epoch_{i}" for i in range(6)],
        inputs=True, function=True, gradients_input=False, gradients_output=False,
    )
