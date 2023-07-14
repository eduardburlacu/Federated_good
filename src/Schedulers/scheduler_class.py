import numpy as np
from typing import List
from src.main import FlowerClient


class Scheduler:
    def __init__(self, clients_set: List[str],pacer: float, exploitation_factor:float):
        self.cids = clients_set
        self.batch = len(clients_set)
        self.clients = [FlowerClient(cid) for cid in clients_set]
        self.epsilon = exploitation_factor
        self.pacer = pacer

    def get_statistical_utility(self, losses, times, T:float, alfa:float = 0.1):
        y = np.sqrt( self.batch * losses.abs2().sum())
        x = y * (T / times)**alfa
        condition = T < times
        return np.where(condition, x, y)

    def update_utility(self, losses, last_involved_round, round_step:int, k:float = 0.1):
        return self.get_statistical_utility(losses) + np.sqrt(k * np.log(round_step)/ last_involved_round)







