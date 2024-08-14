import torch
import torch.nn as nn

from activations.torch import ActivationModule, ReLU, Rational



class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.act1 = ReLU("relu_1")
        self.act2 = Rational(
            init=None,
            degrees=(5, 4),
            device="cpu",
            version="A",
            name="rational_1"
        )

        self.layer1 = nn.Linear(12, 24)
        self.layer2 = nn.Linear(24, 6)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)

        return x
    

def train_epoch(model, criterion, optim, data, labels, batch_size, device):
    n_batches = len(data) // batch_size
    if len(data) % batch_size > 0:
        n_batches += 1

    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, len(data))

        X = data[start:end]
        X = X.to(device)
        Y = labels[start:end]
        Y = Y.to(device)

        pred = model(X)

        loss = criterion(pred, Y)

        optim.zero_grad()
        loss.backward()
        optim.step()


def main(device, num_epochs, batch_size):
    model = Model()
    model = model.to(device)

    data = torch.rand(1000, 12)
    labels = torch.rand(1000, 6) * 10

    criterion = nn.MSELoss()

    n_batches = len(data) // batch_size
    if len(data) % batch_size > 0:
        n_batches += 1

    optim = torch.optim.SGD(model.parameters(), lr=0.001)

    for e in range(num_epochs):
        # Capture
        #  * a snapshot of current function
        #  * inputs in epoch e
        #  * gradients wrt. input and/or output in epoch e
        ActivationModule.create_snapshot(
            snap_name=f"Epoch_{e}",
            label_in="Input", irm="layer",
            function=True, inputs=True, gradients=True,
            max_saves=n_batches  # automatically stop after epoch
        )
        print(f"Epoch: {e}")
        train_epoch(model, criterion, optim, data, labels, batch_size, device)
        
    ActivationModule.export_evolution_graphs(path="./training_example_bar_inputs.gif", snap_names=[f"Epoch_{e}" for e in range(num_epochs)],
                                             function=True, inputs=True)
    ActivationModule.export_evolution_graphs(path="./training_example_bar_grad_in.gif", snap_names=[f"Epoch_{e}" for e in range(num_epochs)],
                                             function=True, gradients_input=True)
    ActivationModule.export_evolution_graphs(path="./training_example_bar_grad_out.gif", snap_names=[f"Epoch_{e}" for e in range(num_epochs)],
                                             function=True, gradients_output=True)

    ActivationModule.export_evolution_graphs(path="./training_example_kde_inputs.gif", snap_names=[f"Epoch_{e}" for e in range(num_epochs)],
                                             function=True, inputs=True, use_kde=True)
    ActivationModule.export_evolution_graphs(path="./training_example_kde_grad_in.gif", snap_names=[f"Epoch_{e}" for e in range(num_epochs)],
                                             function=True, gradients_input=True, use_kde=True)
    ActivationModule.export_evolution_graphs(path="./training_example_kde_grad_out.gif", snap_names=[f"Epoch_{e}" for e in range(num_epochs)],
                                             function=True, gradients_output=True, use_kde=True)


if __name__ == "__main__":
    main("cpu", 5, 100)


        
