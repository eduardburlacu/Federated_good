import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LSTMExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: int = 16,
            num_lstm:int=1,
            bidirectional:bool= True
    ) -> None:

        super(LSTMExtractor, self).__init__(observation_space, features_dim)

        hidden_size = features_dim//2 if bidirectional else features_dim

        self.lstm =torch.nn.LSTM(
            input_size=observation_space.shape[0],
            hidden_size=hidden_size,
            num_layers=num_lstm,
            batch_first=True,
            bidirectional=bidirectional
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        out, (h,c) = self.lstm(observations)
        return out
