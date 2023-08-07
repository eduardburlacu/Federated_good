import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, src_path)
from src.Models.ClassModel import Model
from src.script.parse_config import DatasetNode

class MLR_FEMNIST(Model):
    def __init__(self, cid ) -> None:
        super().__init__(cid)
        self.n_classes = 62
        self.n_inputs = 28 * 28
        self.fc = nn.Linear(self.n_inputs, self.n_classes)

    # pylint: disable-msg=arguments-differ,invalid-name
    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        x = torch.flatten(x,1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x

class MLR_MNIST(Model):
    def __init__(self, cid ) -> None:
        super().__init__(cid)
        self.n_classes = 10
        self.n_inputs = 28 * 28
        self.fc = nn.Linear(self.n_inputs, self.n_classes)

    # pylint: disable-msg=arguments-differ,invalid-name
    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        x = torch.flatten(x,1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x

class MLR_SYNTHETIC(Model):
    def __init__(self, cid) -> None:
        super().__init__(cid)
        self.n_classes = 10
        self.n_inputs = 60
        self.fc = nn.Linear(self.n_inputs, self.n_classes)

    # pylint: disable-msg=arguments-differ,invalid-name
    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        x = self.fc(x)
        x = F.softmax(x,1)
        return x

class LSTM_Shakespeare(Model):
    def __init__(self, cid, *args, **kwargs):
        super().__init__(cid, *args, **kwargs)
        self.n_hidden = 256
        self.n_classes = 80
        self.embedding_size = 8
        self.embedding = nn.Embedding(self.n_classes, self.embedding_size)
        self.lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.n_hidden,
            num_layers=2,
            batch_first=True
        )
        self.fc1 = nn.Linear(
            self.n_hidden,
            self.n_classes
        )

    # pylint: disable-msg=arguments-differ,invalid-name
    def forward(self, x) :
        """Compute forward pass."""
        x = self.embedding(x)
        # x: (sequence_length, num_batches, embedding_size)
        out, (h, c) = self.lstm(x)          # h: (num_layers, num_batches, n_hidden)
        h = h[1,:,:]                        # h: (num_batches, n_hidden)
        x = self.fc1(h)
        x = F.softmax(x,dim=1)
        return x

class LSTM_Sent(Model):
    def __init__(self, cid: str):
        super(LSTM_Sent, self).__init__(cid)
        self.embedding_size = 300
        self.n_hidden = 256
        self.hidden_linear = 30
        self.n_classes = 1
        self.LSTM = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.n_hidden,
            num_layers=2,
            batch_first=True
        )
        self.fc1 = nn.Linear(self.n_hidden, self.hidden_linear)
        self.fc2 = nn.Linear(self.hidden_linear, self.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h, c) = self.lstm(x)          # h: (num_layers, num_batches, n_hidden)
        h = h[1,:,:]                        # h: (num_batches, n_hidden)
        x = self.fc1(h)
        x = F.sigmoid(self.fc2(x))
        return x

def load_models_datanodes(model:str =None , dataset:str = None):
    mapper = [
        (LSTM_Sent, DatasetNode('Sent140')),
        (LSTM_Shakespeare, DatasetNode('Shakespeare')),
        (MLR_MNIST, DatasetNode('MNIST')),
        (MLR_FEMNIST, DatasetNode('FEMNIST')),
    ]
    if model == 'LSTM' and dataset == 'Sent140':
        return mapper[0]
    elif model == 'LSTM' and dataset == 'Shakespeare':
        return mapper[1]
    elif model == 'MLR' and dataset == 'MNIST':
        return mapper[2]
    elif model == 'MLR' and dataset == 'FEMNIST':
        return mapper[3]
    elif model is None and dataset is None:
        return mapper
    else: raise AttributeError

def load_datanodes(dataset:str = None):
    mapper =[
        DatasetNode('Sent140'),
        DatasetNode('Shakespeare'),
        DatasetNode('MNIST'),
        DatasetNode('FEMNIST')
    ]
    if dataset == 'Sent140': return mapper[0]
    elif dataset == 'Shakespeare': return mapper[1]
    elif dataset == 'MNIST': return mapper[2]
    elif dataset == 'FEMNIST': return mapper[3]
    else: raise AttributeError


