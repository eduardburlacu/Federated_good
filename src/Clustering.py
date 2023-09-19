import random
from typing import List, Tuple, Dict, Optional
from math import ceil


class PriorityQueue:
    def __init__(self,
                 batch_size:int,
                 straggler_cids:Optional[List[str]]= None,
                 unselected_cids: Optional[List[str]]= None,
                 capacity:Dict[str,float]= None,
                 datasize:Dict[str,int]= None,
                 ):
        self.batch_size = batch_size
        self.straggler_queue = []
        self.follower_queue = []
        if capacity and datasize:
            self.capacity = capacity.copy()
            self.datasize = datasize.copy()
            self.straggler_queue  = sorted(straggler_cids, key=lambda x: capacity[x], reverse=True)  #highest capacity to be solved first
            self.follower_queue = sorted(unselected_cids,key=lambda x: capacity[x]/datasize[x], reverse=True) # sort as if the dataset has size 1 initially
            for cid in unselected_cids:
                self.datasize[cid]= 0
        else:
            raise AttributeError("Scheduler not given all parameters.")

    def __len__(self)->int:
        return len(self.straggler_queue)

    def __repr__(self):
        return f"Stragglers:{repr(self.straggler_queue)} \n Unselected: {repr(self.follower_queue)}"

    def peek(self, straggler=True)-> Tuple[str,float]:
        "Show the top priority straggler(if True) or follower(if False)"
        if straggler:
            return self.straggler_queue[0], self.capacity[self.straggler_queue[0]]
        else:
            return self.follower_queue[0], self.capacity[self.follower_queue[0]]
    def dequeue(self)-> Tuple[str,str]:
        "Get the highest priority mapping"
        if len(self.straggler_queue)>0:
            straggler = self.straggler_queue.pop(0)
        else: straggler=""

        if len(self.follower_queue)>0:
            follower = self.follower_queue.pop(0)
        else: follower=""

        return straggler,follower

    def enqueue(self, follower:str, added_len_dataset:int):
        #update the capacity
        self.capacity[follower] *= (ceil( added_len_dataset + self.datasize[follower]/self.batch_size)/ max(1,ceil(self.datasize[follower]/self.batch_size)))
        #update the total datasize
        self.datasize[follower] += added_len_dataset

        if len(self.straggler_queue)>0:
            tmp = 0
            while self.datasize[self.follower_queue[tmp]]< self.datasize[follower]:
                tmp+=1
                if tmp==len(self.follower_queue):
                    break
        else: tmp=1

        self.follower_queue.insert(tmp-1, follower)

    def schedule(self):
        jobs={}
        mappings={}
        while len(self.straggler_queue)>0: # stop when all stragglers are assigned an offloading behaviour
            straggler, follower = self.dequeue()
            if straggler=="": break
            #check if it is better to leave the computation there
            if self.capacity[straggler]>=self.capacity[follower]:
                if follower in jobs:
                    jobs[follower] +=1
                else:
                    jobs[follower] =1
                mappings[straggler]=follower
                added_len = self.datasize[straggler]
            else: added_len= 0

            self.enqueue(
                follower=follower,
                added_len_dataset=added_len,
            )
        return jobs, mappings

class SchedulerPriority:
    def __init__(self, batch_size:int,):
        self.batch_size = batch_size

    def _cluster(self,
                 selected_cids:List[str],
                 unselected_cids:List[str],
                 stragglers:Dict[str, int],
                 ):
        # Cluster Devices
        self.selected_stragglers=[]
        self.unselected_cids=[]
        self.selected_stragglers=[cid for cid in selected_cids if stragglers[cid]==1]
        self.unselected_cids = [
            cid for cid in unselected_cids
            if stragglers[cid]==0
        ]   #filter out the stragglers not selected this round

    def get_mappings(self,
                     selected_cids:List[str],
                     unselected_cids:List[str],
                     capacity:Dict[str, float],
                     stragglers:Dict[str, int],
                     datasize:Dict[str,int],
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
        :return: Tuple of 3 components:
                    -- jobs - Dictionary for followers allocated as
                    cid--> count of stragglers assigned to that.
                    -- mappings - Dictionary mapping straggler cid -->
                    follower cid
                    -- clients -  A list of cids containg all the participants
                    to FL round, i.e. basic clients, stragglers, and followers
        '''

        self._cluster(
            selected_cids=selected_cids,
            unselected_cids=unselected_cids,
            stragglers=stragglers,
        )
        queue = PriorityQueue(
            batch_size=self.batch_size,
            straggler_cids=self.selected_stragglers,
            unselected_cids=self.unselected_cids,
            capacity=capacity,
            datasize=datasize
        )
        jobs, mappings = queue.schedule()
        clients = [*jobs.keys(), *selected_cids]

        return jobs, mappings, clients

class SchedulerRoundRobin:
    def __init__(self):
        self.selected_cids = []
        self.unselected_cids = []
        self.capacity = []

    def _update(self,
                selected_cids:List[str],
                unselected_cids:List[str],
                capacity:Dict[str, float],
                stragglers:Dict[str, int],
                ):

        self.selected_cids = selected_cids.copy()
        self.unselected_cids = [cid for cid in unselected_cids if stragglers[cid]==0]
        self.capacity = capacity

    def get_mappings(self,
                     selected_cids:List[str],
                     unselected_cids:List[str],
                     capacity:Dict[str, float],
                     stragglers:Dict[str, int],
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

        self.selected_cids.sort(key= lambda x: capacity[x], reverse = True)
        self.unselected_cids.sort(key= lambda x: capacity[x], reverse = True)
        jobs={}
        mappings={}
        clients= self.selected_cids
        lim = len(self.unselected_cids)
        if lim > 0:
            for idx, cid in enumerate(self.selected_cids):
                if stragglers[cid]==1:
                    mappings[cid] = self.unselected_cids[ idx%lim ]
                    if self.unselected_cids[idx%lim] not in jobs:
                        jobs[self.unselected_cids[idx % lim]] = 1
                    else:
                        jobs[self.unselected_cids[idx % lim]]+=1

            clients=[ *jobs.keys(), *self.selected_cids]

        return jobs, mappings, clients

if __name__=="__main__":

    from straggler_schedule import get_straggler_schedule
    stragglers_mat, computation_fracs = get_straggler_schedule(
        num_clients=1000,
        num_rounds=20,
        stragglers_frac=0.8,
        type="constant",
    )
    init_stragglers = {str(cid): bool(stragglers_mat[cid, 0]) for cid in range(1000)}
    clients = [str(n) for n in range(1000)]
    datasize = {cid: random.randint(10, 1000) for cid in clients}
    capacity={cid: float(cid) for cid in clients}
    scheduler = SchedulerPriority(10)
    shit_scheduler = SchedulerRoundRobin()
    jobs,mappings,clients=scheduler.get_mappings(
        selected_cids=clients[:700],
        unselected_cids=clients[700:],
        capacity=capacity,
        stragglers=init_stragglers,
        datasize=datasize
    )
    shit_jobs,shit_mappings,shit_clients=shit_scheduler.get_mappings(
        selected_cids=clients[:700],
        unselected_cids=clients[700:],
        capacity=capacity,
        stragglers=init_stragglers,
    )
    print(f"SELECTED CLIENTS ARE{clients} AND \n JOBS ARE: {jobs} AND \n MAPPINGS ARE {mappings}")
    print(f"SELECTED CLIENTS ARE{shit_clients} AND \n JOBS ARE: {shit_jobs} AND \n MAPPINGS ARE {shit_mappings}")

# lim = len(self.unselected_cids)
# follower_idx = 0
# for selected_idx,cid in enumerate(self.selected_cids):
#    if stragglers[cid]==1:
#        if follower_idx<lim:
#            mappings[cid]

# if lim > 0:
#    for idx, cid in enumerate(self.selected_cids):
#        if stragglers[cid]==1:


#            mappings[cid] = self.unselected_cids[ idx%lim ]
#            if self.unselected_cids[idx%lim] not in jobs:
#                jobs[self.unselected_cids[idx % lim]] = 1
#            else:
#                jobs[self.unselected_cids[idx % lim]]+=1
#
#    clients=[ *jobs.keys(), *self.selected_cids]