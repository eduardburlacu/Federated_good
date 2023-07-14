import torch
import torch.nn as nn
from torch.nn import functional as F
from src.Models.ClassModel import Model

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

class LSTM_Sent(Model):
    def __init__(self, cid: str):
        super(LSTM_Sent, self).__init__(cid)
        self.embedding_size = 300
        self.n_hidden = 256
        self.LSTM = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.n_hidden,
            num_layers=2,
            batch_first=True
        )
        self.fc1 = nn.Linear(2*self.n_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h, c) = self.LSTM(x)
        h = h.permute(1,0,2)
        h = h.view(-1, 2 * self.n_hidden)
        x = self.fc1(h)
        x = F.sigmoid(x)
        return x

