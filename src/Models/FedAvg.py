import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, src_path)
from src.Models.ClassModel import Model
from src.script.parse_config import DatasetNode
class CNN_MNIST(Model):
    def __init__(self, cid: str, *args, **kwargs):
        super(CNN_MNIST, self).__init__(cid, *args, **kwargs)
        self.conv1 = nn.Conv2d(1, 32, 5, padding='same') #  32*28*28
        self.pool1 = nn.MaxPool2d(2, 2)                  #  32*14*14
        self.conv2 = nn.Conv2d(32,64, 5, padding='same') #  64*14*14
        self.pool2 = nn.MaxPool2d(2,2)                   #  64*7*7
        self.fc1 = nn.Linear(64* 7* 7, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = x.view(-1, 64* 7* 7)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

class CNN_CIFAR(Model):
    def __init__(self, cid: str, *args, **kwargs):
        super(CNN_CIFAR, self).__init__(cid, *args, **kwargs)
        self.conv1 = nn.Conv2d(3, 32, 3, padding='valid') #  32*30*30
        self.pool1 = nn.MaxPool2d(2, 2)                   #  32*15*15
        self.conv2 = nn.Conv2d(32,64, 3, padding='valid') #  64*13*13
        self.pool2 = nn.MaxPool2d(2,2)                    #  64* 6* 6
        self.conv3 = nn.Conv2d(64, 64 ,3, padding='valid')#  64* 4* 4
        self.fc1 = nn.Linear(64* 4* 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x)
        x = x.view(-1, 64* 4* 4)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x
class MLP_MNIST(Model):
  def __init__(self, cid:str, input_size:int = 28*28*1, *args, **kwargs):
    super(MLP_MNIST, self).__init__(cid, *args, **kwargs)
    self.fc1 = nn.Linear(input_size, 200)
    self.fc2 = nn.Linear(200,200)
    self.fc3 = nn.Linear(200, 10)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x.view(-1, 28*28*1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.softmax(self.fc3(x), dim=1)
    return x

class LSTM_Shakespeare(Model):
    def __init__(self, cid, *args, **kwargs):
        super().__init__(cid, *args, **kwargs)
        self.n_hidden = 256
        self.n_classes = 80
        self.embedding_size = 8
        self.embedding = nn.Embedding(80, 8)

        self.lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.n_hidden,
            num_layers=2,
            batch_first=True
        )
        self.fc1 = nn.Linear(
            self.n_hidden * 2,
            self.n_classes
        )

    # pylint: disable-msg=arguments-differ,invalid-name
    def forward(self, x) :
        """Compute forward pass."""
        x = self.embedding(x)
        # x: (sequence_length, num_batches, embedding_size)
        out, (h, c) = self.lstm(x)          # h: (num_layers, num_batches, n_hidden)
        h = torch.permute( h, (1, 0, 2))    # h: (num_batches, num_layers, n_hidden)
        h = h.view(-1, 2* self.n_hidden) # h: (num_batches, 2 * n_hidden)
        x = self.fc1(h)
        x = F.softmax(x,dim=1)
        return x

class LSTM_Large(Model):
    def __init__(self, cid,*args,**kwargs):
        super(LSTM_Large, self).__init__(cid,*args, **kwargs)
        self.n_hidden = 256
        self.n_classes = 10000
        self.embedding_size = 192
        self.embedding = nn.Embedding(self.n_classes, self.embedding_size)

        self.lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.n_hidden,
            num_layers=1,
            batch_first=True
        )
        self.fc1 = nn.Linear(
            self.n_hidden,
            self.n_classes )

    # pylint: disable-msg=arguments-differ,invalid-name
    def forward(self, x) :
        """Compute forward pass."""
        x = self.embedding(x)               # x: (num_batches, sequence_length, embedding_size)
        out, (h, c) = self.lstm(x)          # h: (num_layers, num_batches, n_hidden)
        h = torch.permute( h, (1, 0, 2))    # h: (num_batches, num_layers, n_hidden)
        h = h.view(-1, self.n_hidden)       # h: (num_batches, n_hidden)
        x = self.fc1(h)
        x = F.softmax(x,dim=1)
        return x
def load_models_datanodes(model:str =None , dataset:str = None, iid:bool = True):
    mapper = [
        (CNN_MNIST, DatasetNode('MNIST',iid)),
        (MLP_MNIST, DatasetNode('MNIST', iid)),
        (CNN_CIFAR, DatasetNode('CIFAR10', iid)),
    ]
    if   model == 'CNN' and dataset=='MNIST': return mapper[0]
    elif model == 'MLP' and dataset=='MNIST': return mapper[1]
    elif model == 'CNN' and dataset=='CIFAR10': return mapper[2]
    elif model is None and dataset is None: return mapper
    else: raise AttributeError

def load_datanodes(dataset:str = None):
    mapper = [
        DatasetNode('MNIST'),
        DatasetNode('CIFAR10'),
        DatasetNode('Shakespeare'),
    ]
    if   dataset=='MNIST': return mapper[0]
    elif dataset=='CIFAR10': return mapper[1]
    elif dataset=='Shakespeare': return mapper[2]
    elif dataset is None: return mapper
    else: raise AttributeError
