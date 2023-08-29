from typing import Dict, List, Optional, Set, Tuple
from logging import INFO

import random

import flwr
from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager
from flwr.server.criterion import Criterion
from flwr.server.client_proxy import ClientProxy

from src.Clustering import Scheduler


class OffloadClientManager(SimpleClientManager):
    def __init__(self, scheduler:Scheduler = None):
        super(OffloadClientManager, self).__init__()
        self.scheduler = scheduler

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
        with_followers:bool = True,
        stragglers:Set[str]=None,
        capacity:Dict[str, float] = None
    ) -> Tuple[
        List[ClientProxy],
        Dict[str,str]
    ]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return [],{}

        random.shuffle(available_cids)
        sampled_cids = available_cids[:num_clients]
        unsampled_cids = available_cids[num_clients:]


        if with_followers:      #Include followers
            jobs, selected_cids= self.scheduler.get_mappings(
                selected_cids=sampled_cids,
                unselected_cids=unsampled_cids,
                capacity=capacity,
                stragglers=stragglers,
                priority_sort=with_followers,
            )
            return [self.clients[cid] for cid in selected_cids], jobs

        else: return [self.clients[cid] for cid in sampled_cids], {}
