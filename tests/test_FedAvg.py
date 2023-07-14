from src.Models.FedAvg import *

net1 = MLP_MNIST('1')
total_params1 = sum(p.numel() for p in net1.parameters())
assert total_params1==199210

net2 = CNN_MNIST('2')
total_params2 = sum(p.numel() for p in net2.parameters())
assert total_params2==1663370

net3 = CNN_CIFAR('3')
total_params3 = sum(p.numel() for p in net3.parameters())
assert total_params3 == 122570

net4 = LSTM_Shakespeare('4')
print(net4)
total_params4 = sum(p.numel() for p in net4.parameters())
#assert total_params4==866578


            #    batch seq-length vocab_size
x = torch.randint([7, 5, 10])
x= nn.Embedding(10,19)(x)
print(x)

'''
net5=LSTM_Large('5')
print(net5)
total_params5 = sum(p.numel() for p in net4.parameters())
print(total_params5)

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

'''