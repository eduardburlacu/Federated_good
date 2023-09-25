"""
Custom Gym environment for optimally
splitting the Neural Networks in SFD
"""

from typing import Any, SupportsFloat, Dict, TypeVar, Union
import numpy as np
from numpy import log,exp, sqrt
import gymnasium as gym
from gymnasium import spaces
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

def get_action_space(num_layers:int, seed:int=0):
    return spaces.Discrete(
        n=num_layers,
        seed=seed
    )
def get_observ_space(num_observations:int):
    return spaces.Box(
        low=[0.0 for _ in range(num_observations)],
        high=[float("inf") for _ in range(num_observations)],
        shape=(num_observations,),
        dtype=np.float32,
        seed=0
    )

class FL_Env(gym.Env):
    metadata = {'render.modes': ['human']}
    reward_range = (-float("inf"), 0.0)

    def __init__(self, FL_ROUNDS:int):
        self.rounds = FL_ROUNDS

    @staticmethod
    def reward_fn(train_time: float, timeout: Union[int, float]):
        if train_time >= timeout:
            return - train_time
        else:
            return log(1 - exp(-sqrt(train_time)), 0.995)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        pass

    def reset(
        self,
        *,
        seed: int = None,
        options: Dict[str, Any] = None,
    ) -> tuple[ObsType, Dict[str, Any]]:
        pass


    def close(self):
        pass