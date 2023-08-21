import torch
import torch.nn as nn
import torch.nn.functional as F


class base_net(nn.Module):
    def __init__(self, input_size, dim1, dim2, output_size):
        self.fc1 = nn.Linear(input_size,dim1)
        self.fc2 = nn.Linear(dim1,dim2)
        self.fc3 = nn.Linear(dim2,output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

class Attention(nn.Module):
    def __init__(self, input_size:int,dim1:int, dim2:int, embed_dim:int, num_heads:int =1):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.Query = base_net(input_size,dim1,dim2,embed_dim)
        self.Key = base_net(input_size,dim1,dim2,embed_dim)
        self.Value = base_net(input_size,dim1,dim2,embed_dim)

    def forward(self, *inputs)-> torch.Tensor:
        obs_k, obs_v = inputs
        q = self.Query(obs_k)
        k = self.Key(obs_k)
        v = self.Value(obs_v)
        x =  nn.MultiheadAttention(self.embed_dim,
                                   self.num_heads,
                                   )
        return x


