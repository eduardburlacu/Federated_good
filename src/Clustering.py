import random
from typing import List, Tuple, Dict, Set

class ClusteringModule:
    def __init__(self,
                 flops: List[int],
                 r_low:float,
                 r_high:float,
                 ):

        self.flops_all = flops
        self.client_number_all = len(flops)
        self.r_low = r_low
        self.r_high = r_high

        self.flops = []
        self.client_number = 0
        self.tier_high_cid= []
        self.tier_mid_cid = []
        self.tier_low_cid = []


    def update(self, available_cids:List[str]):
        self.flops = [self.flops_all[int(cid)] for cid in available_cids]
        self.client_number = len(self.flops)

    def get_tiers(self, importance_sort:bool =True):
        x = sorted(self.flops)
        th_high = x[int(self.r_high * self.client_number)]
        th_low  = x[int(self.r_low  * self.client_number)]
        print(f"th_high:{th_high} and th_low:{th_low}")

        for idx, flop in enumerate(self.flops):
            if flop>= th_high:
                self.tier_high_cid.append((idx,flop))
            elif flop >= th_low:
                self.tier_mid_cid.append((idx,flop))
            else:
                self.tier_low_cid.append((idx,flop))
        if importance_sort:
            self._importance_sort()

    def _importance_sort(self):
        self.tier_high_cid.sort(key= lambda x:x[1], reverse=True)
        self.tier_low_cid.sort(key=lambda x: x[1], reverse=False)

    def print(self):
        print("Tier high queue-->",self.tier_high_cid)
        print("Tier mid queue-->" ,self.tier_mid_cid)
        print("Tier low queue-->" ,self.tier_low_cid)

    def straggler_cids(self):pass


class Scheduler:
    def __init__(self, schedule:str='round_robin',):

        self.schedule = schedule
        self.selected_cids = []
        self.unselected_cids = []

        self.capacity = []
        self.jobs = {}

    def _update(self,
                selected_cids:List[str],
                unselected_cids:List[str],
                capacity:Dict[str, float],
                stragglers:Set
                ):

        self.selected_cids = selected_cids.copy()
        self.unselected_cids = [cid for cid in unselected_cids if stragglers[cid]==0]
        self.capacity = capacity
        self.jobs= {}

    def get_mappings(self,
                     selected_cids:List[str],
                     unselected_cids:List[str],
                     capacity:Dict[str, float],
                     stragglers:Set[str],
                     priority_sort=True
                     ) -> Tuple[
        Dict[str, int],
        Dict[str,str],
        List[str]
    ]:
        '''
        :param selected_cids: List containing all the cids
         selected for the current round.
        :param unselected_cids: List containing all the cids
         not selected for the current round.
        :param capacity: Dictionary cid --> capacity[cid]
         showing the performance of the client, which can change
         over training.
        :param stragglers: Set containing  all stragglers in
         the selected round by the respective client manager.
        :param priority_sort:   Prioritise stragglers with low capacity
         and suggest first unselected devices with superior capacity
        :return: Tuple of 3 components:
                    -- jobs - Dictionary for followers allocated as
                    cid--> count of stragglers assigned to that.
                    -- mappings - Dictionary mapping straggler cid -->
                    follower cid
                    -- clients -  A list of cids containg all the participants
                    to FL round, i.e. basic clients, stragglers, and followers
        '''

        self._update(
            selected_cids=selected_cids,
            unselected_cids=unselected_cids,
            capacity=capacity,
            stragglers=stragglers,
        )
        if priority_sort:
            self.selected_cids.sort(key= lambda x: capacity[x], reverse = False)
            self.unselected_cids.sort(key= lambda x: capacity[x], reverse = True)
        jobs={}
        mappings={}
        clients= self.selected_cids
        if self.schedule == "round_robin":
            lim = len(self.unselected_cids)
            if lim > 0:
                for idx, cid in enumerate(self.selected_cids):
                    if cid in stragglers:
                        mappings[cid] = self.unselected_cids[ idx%lim ]
                        if self.unselected_cids[idx%lim] not in jobs:
                            jobs[self.unselected_cids[idx % lim]] = 1
                        else:
                            jobs[self.unselected_cids[idx % lim]]+=1

                #for follower, straggler in zip(jobs.keys(), self.selected_cids):
                #    clients.append(follower)
                #    clients.append(straggler)
                clients=[ *jobs.keys(), *self.selected_cids,]

        return jobs, mappings, clients


if __name__=='__main__':
    '''
    flops = [15*n - n**2 for n in range(15)]
    c = ClusteringModule(flops, 0.25, 0.75)
    c.get_tiers()
    c.print()
    
    '''
    scheduler = Scheduler(
        r_low=0.1,
        r_high=0.2
    )
    selected = '12 3 4 2 1 6 7 9'.split(sep=' ')
    unselected = '13 14 19 20'.split(sep=' ')
    capacity={ x: random.random() for x in [*selected, *unselected]}
    results = scheduler.get_mappings(
        selected_cids=selected,
        unselected_cids=unselected,
        capacity = capacity,
        stragglers= set(selected[:4])
    )
    for result in results:
        print(result)

