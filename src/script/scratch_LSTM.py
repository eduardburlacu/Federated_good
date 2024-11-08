import torch
from torch import nn, Tensor
from torch.nn import functional as F

class LSTM(nn.Module):
    def __init__(self, n_hidden, n_classes, cid) -> None:
        super().__init__(cid)
        self.n_hidden = n_hidden  # 256 in FedProx and LEAF
        self.n_classes = n_classes
        self.embedding_size = 300
        self.fc1 = nn.Linear(
            self.n_hidden,
            self.n_classes
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.n_hidden,
            num_layers=2,
            batch_first=True
        )

    # pylint: disable-msg=arguments-differ,invalid-name
    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        # x: (sequence_length, num_batches, embedding_size)
        _, (h, _) = self.lstm(x)
        # h:  (num_layers, num_batches, n_hidden)
        h = h[1,:,:]
        # h:  (num_batches, n_hidden)
        x = self.fc1(h)
        return x

class LSTM_Shakespeare(nn.Module):
    def __init__(self, n_hidden, n_classes, cid) -> None:
        super().__init__(cid)
        self.n_hidden = 256  # 256 in FedProx and LEAF
        self.n_classes = 80
        self.embedding_size = 8
        self.fc1 = nn.Linear(
            self.n_hidden * 2,
            self.n_classes
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.n_hidden,
            num_layers=2,
            batch_first=True
        )
        self.embedding = nn.Embedding(self.n_classes, self.embedding_size)

    # pylint: disable-msg=arguments-differ,invalid-name
    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        # x: (num_batches, sequence_length)
        x = self.embedding(x)
        # x: (num_batches, sequence_length, embedding_size)
        out, (h, c) = self.lstm(x)
        # h:  (num_layers, num_batches, n_hidden)
        # h:  (num_batches, n_hidden)
        x = self.fc1(h)
        return x


if __name__=='__main__':
    layer = nn.LSTM(
            input_size=80,
            hidden_size=256,
            num_layers=2,
            batch_first=True)
    inp = torch.randn([17,10,80])
    out,(hn,cn) = layer(inp)
    print(out.shape,'\n',hn.shape, '\n', cn.shape)



