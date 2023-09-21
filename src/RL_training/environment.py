import os
import sys
import flwr as fl
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

#-------Insert main project directory so that we can resolve the src imports-------
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, src_path)

from src.run import main

class Environment():

    def __init__(self, *args, **kwargs):
        self.state_space =None
        self.action_space = None
        self.state = None
        self.action = None
        self.reward = 0.

    def start_episode(self):
        #update reward
        self.reward = - main()




