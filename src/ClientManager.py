from typing import Dict, List, Optional, Tuple, Union
from logging import INFO
import random

from flwr.common.typing import GetPropertiesIns, GetPropertiesRes
from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager
from flwr.server.criterion import Criterion
from flwr.server.client_proxy import ClientProxy

from src import SEED
from src.Clustering import SchedulerPriority

random.seed(SEED)

class OffloadClientManager(SimpleClientManager):
    def __init__(self, batch_size:Union[str,int,float]):
        super(OffloadClientManager, self).__init__()
        self.scheduler = SchedulerPriority(int(batch_size))
        self.query = {"curr_round":1}

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,

        with_followers:Optional[bool] = False,
        stragglers:Optional[Dict[str,int]]=None,
        capacities:Optional[Dict[str,float]]=None,
        datasizes: Optional[Dict[str,int]]=None,
        ports:Optional[Dict[str,int]]=None,
    ) -> Union[
        List[ClientProxy],
        Tuple[
        List[ClientProxy],
        List[str],
        Dict[str, int],
        Dict[str, int]
        ]
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

        if with_followers:  #Include followers
            query = (stragglers is None) or (capacities is None)or (datasizes is None) or (ports is None)
            if query:
                raise AttributeError("Client Manager did not get enoough args")

            #Udate initial_capacities for 'round 0'
            #if max(capacities.values())==0.:
            #    for cid,client_proxy in self.clients.items():
            #        res:GetPropertiesRes = client_proxy.get_properties(GetPropertiesIns({}),timeout=10)
            #        capacities[cid] = res.properties["capacity"]
            jobs, mappings, selected_cids= self.scheduler.get_mappings(
                selected_cids= sampled_cids,
                unselected_cids= unsampled_cids,
                capacity= capacities,
                stragglers= stragglers,
                datasize=datasizes,
            )
            port_conf = {
                straggler: ports[follower]
                for straggler, follower in mappings.items()
            }

            return [self.clients[cid] for cid in selected_cids], selected_cids, jobs, port_conf

        else: # Necessary for Flwr library integration + using classic FedProx
            return [self.clients[cid] for cid in sampled_cids]
