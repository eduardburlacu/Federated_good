import torch
import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, src_path)
from src.RL.policy import *

def test_policy():
    vector = torch.rand((17, 3))
    print(vector)
    model = LSTMExtractor(gym.spaces.Box(np.array([0.0 for i in range(3)]), np.array([1.0 for j in range(3)])))
    print(model)
    out = model(vector)
    print(out)
    assert torch.Size((17,16))==out.shape