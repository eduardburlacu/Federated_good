"""
Custom Gym environment for optimally
splitting the Neural Networks in SFD
"""

from typing import Any, SupportsFloat, Dict, List, TypeVar, Union
import numpy as np
from numpy import log, exp, sqrt
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import RenderFrame
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
from src.RL import LOG_BASE, SEED
def get_action_space(num_layers:int, seed:int=SEED):
    return spaces.Discrete(
        n=num_layers,
        seed=seed
    )
def get_observ_space(num_observations:int, seed:int=SEED):
    return spaces.Box(
        low=np.asarray([0.0 for _ in range(num_observations)]),
        high=np.asarray([float("inf") for _ in range(num_observations)]),
        shape=(num_observations,),
        dtype=np.float32,
        seed=seed
    )

class EnvFL(gym.Env):
    metadata = {'render.modes': ['human']}
    reward_range = (-float("inf"), float("inf"))

    def __init__(
            self,
            num_rounds:int = 100,
            num_split_points:int=7,
            seed:int=SEED,
    ):
        super(EnvFL, self).__init__()

        self.num_rounds = num_rounds

        #gym API attrs
        self.action_space = spaces.Discrete(
            n=num_split_points,
            seed=seed
        )
        self.observation_space = spaces.Box(
            low = 0.0,
            high= float("inf"),
            shape=(4,), # Each row corresponds to: (strag_cap,strag_compl_cap,fol_cap,fol_compl_cap)
            dtype=np.float32,
            seed=seed,
        )
        self.reward_range = (
            -float("inf"),
            float("inf")
        )
        self.np_random = np.random.default_rng(seed=seed)

    @staticmethod
    def reward_fn(
            train_time: float,
            timeout: Union[int, float],
            log_base:float=LOG_BASE
    ):
        if train_time >= timeout:
            return - train_time
        else:
            return log(1 - exp(-sqrt(train_time)), log_base)

    def reset(
        self,
        *,
        seed: int = None,
        options: Dict[str, Any] = None,
    ) -> tuple[
        ObsType,
        Dict[str, Any]
    ]:
        pass

    def step(
        self,
        action: ActType
    ) -> tuple[
        ObsType,
        SupportsFloat,
        bool,
        bool,
        dict[str, Any]
    ]:
        pass

    def render(self) -> Union[
        RenderFrame,
        List[RenderFrame],
        None
    ]:
        pass

    def close(self):
        pass