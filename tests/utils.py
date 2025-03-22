import torch
import torch.nn as nn
import torch.optim as optim
import time
import torchvision
import torchvision.transforms as transforms
from activations.torch.classic_activations import GLU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


class TestModel(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.activation = activation_fn
        self._fc1_out = 128
        self._use_glu = isinstance(activation_fn, GLU)

        try:
            device_for_init = next(activation_fn.parameters()).device
        except StopIteration:
            device_for_init = device

        # Temporary inference for shape
        with torch.no_grad():
            conv1 = nn.Conv2d(3, 32, 3, 1, 1).to(device_for_init)
            conv2 = nn.Conv2d(32, 64, 3, 1, 1).to(device_for_init)
            pool = nn.MaxPool2d(2, 2).to(device_for_init)
            dummy_input = torch.zeros(1, 3, 32, 32, device=device_for_init)
            x = pool(activation_fn(conv1(dummy_input)))
            x = pool(activation_fn(conv2(x)))
            flatten_dim = x.view(1, -1).shape[1]

        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self._flatten_dim = flatten_dim

        if self._use_glu:
            self.fc1 = nn.Linear(flatten_dim, self._fc1_out * 2)
            self.fc2 = nn.Linear(self._fc1_out, 10)
        else:
            self.fc1 = nn.Linear(flatten_dim, self._fc1_out)
            self.fc2 = nn.Linear(self._fc1_out, 10)

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


def train_and_evaluate(activation_fn, activation_name, num_epochs=1):
    model = TestModel(activation_fn).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for _ in range(num_epochs):
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total


def train_and_evaluate_perf(activation_fn, activation_name, num_epochs=5):
    model = TestModel(activation_fn).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    train_duration = time.time() - start_time

    # Final accuracy on train
    correct_train, total_train = 0, 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
    train_acc = 100.0 * correct_train / total_train

    # Final accuracy on test
    correct_test, total_test = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total_test += labels.size(0)
            correct_test += predicted.eq(labels).sum().item()
    test_acc = 100.0 * correct_test / total_test

    return train_duration, train_acc, test_acc

def compare_gradients(act_a, act_b, input_tensor, target_tensor):
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    target_tensor = target_tensor.clone().detach()

    # Clone inputs for both activations
    input_a = input_tensor.clone().detach().requires_grad_(True)
    input_b = input_tensor.clone().detach().requires_grad_(True)

    # Forward
    output_a = act_a(input_a)
    output_b = act_b(input_b)

    # Same loss
    loss_fn = nn.MSELoss()
    loss_a = loss_fn(output_a, target_tensor)
    loss_b = loss_fn(output_b, target_tensor)

    # Backward
    grad_input_a = torch.autograd.grad(loss_a, input_a, retain_graph=True)[0]
    grad_input_b = torch.autograd.grad(loss_b, input_b, retain_graph=True)[0]

    return torch.allclose(grad_input_a, grad_input_b, atol=1e-5, rtol=1e-4)

