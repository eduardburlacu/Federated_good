import numpy as np
import flwr as fl
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import List
import os
import importlib.util
from src import PATH_src

def set_random_seed(seed: int):
    random.seed(1+seed)
    np.random.seed(12 + seed)
    torch.manual_seed(123 + seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123 + seed) # Set seed for CUDA if available

def importer(filename:str):
    module_path = os.path.join(PATH_src['Models'], filename+'.py') # Specify path to the file you want to import from
    spec = importlib.util.spec_from_file_location(filename, module_path)  # Load the module from the specified path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class Net(nn.Module):
    '''
    Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')
    borrowed from Pytorch quickstart example
    '''
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# borrowed from Pytorch quickstart example
def train(net, trainloader, epochs, device):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

# borrowed from Pytorch quickstart example
def test(net, testloader, device):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

def get_params(model) -> List[np.ndarray]:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_params(model, params: List[np.ndarray]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
