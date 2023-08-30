from typing import Dict, List, Optional, Tuple, Union
from logging import INFO

import random
from flwr.common.typing import GetPropertiesIns, GetPropertiesRes
from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager
from flwr.server.criterion import Criterion
from flwr.server.client_proxy import ClientProxy

from src.Clustering import Scheduler


class OffloadClientManager(SimpleClientManager):
    def __init__(self, schedule:str="round_robin"):
        super(OffloadClientManager, self).__init__()
        self.scheduler = Scheduler(schedule=schedule)
        self.query = {"curr_round":1}

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
        with_followers:bool = False,
    ) -> Union[
        Tuple[
        List[ClientProxy],
        List[str],
        Dict[str, int],
        Dict[str, int]
        ],
        List[ClientProxy]
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
            self.query["curr_round"] += 1
            if with_followers:
                return [],[],{},{}
            else:
                return []
        random.shuffle(available_cids)
        sampled_cids = available_cids[:num_clients]
        unsampled_cids = available_cids[num_clients:]

        if with_followers:      #Include followers
            capacity = {}
            stragglers = set()
            res=[None]* len(self.clients)
            for cid in available_cids:
                res[int(cid)] = self.clients[cid].get_properties(GetPropertiesIns(self.query), timeout=None).properties
                if res[int(cid)]["straggler"] == 1:
                    stragglers.add(cid)
                capacity[cid] = res[int(cid)]["capacity"]
            jobs, mappings, selected_cids= self.scheduler.get_mappings(
                selected_cids= sampled_cids,
                unselected_cids= unsampled_cids,
                capacity= capacity,
                stragglers= stragglers,
                priority_sort= with_followers,
            )
            ports = {
                straggler: int(res[int(follower)]["port"])
                for straggler, follower in mappings.items()
            }
            del res
            self.query["curr_round"] += 1
            return [self.clients[cid] for cid in selected_cids], selected_cids,jobs, ports

        else:
            #Necessary for Flwr library parameter initiation + using classic FedProx
            return [self.clients[cid] for cid in sampled_cids]
