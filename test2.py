import os
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from activations.torch.learnable_activations.rationals.cuda_impl import RationalCUDA
from activations.torch.learnable_activations.rationals import Rational

os.environ['CUDA_VISIBLE_DEVICES']='2'

torch.manual_seed(42)
random.seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CIFAR-10 dataset
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
        super(TestModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.activation = activation_fn

        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Adjusted for correct dimensions
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))

        x = torch.flatten(x, start_dim=1)  # Fixed dynamic flattening
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


# Function to train and evaluate models
def train_and_evaluate(activation_fn, activation_name, num_epochs=5):
    model = TestModel(activation_fn).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Timing & Profiling
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total
        avg_loss = running_loss / len(trainloader)

        print(f"[{activation_name}] Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Accuracy={train_acc:.2f}%")

    end_time = time.time()
    print(f"[{activation_name}] Training completed in {end_time - start_time:.2f}s")

    # Evaluate on test data
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

    test_acc = 100.0 * correct / total
    print(f"[{activation_name}] Test Accuracy: {test_acc:.2f}%\n")

# Run tests for different activations
train_and_evaluate(nn.ReLU(), "ReLU")
train_and_evaluate(nn.SiLU(), "SiLU")
train_and_evaluate(Rational(), "RationalTORCH")
train_and_evaluate(RationalCUDA(), "RationalCUDA")
