from typing import List, Tuple, Union, Dict, Set

class Clustering_Module:
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
        x = sorted(flops)
        th_high = x[int(self.r_high * self.client_number)]
        th_low  = x[int(self.r_low  * self.client_number)]
        print(f"th_high:{th_high} and th_low:{th_low}")

        for idx, flop in enumerate(flops):
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
    def __init__(self, r_low:float,r_high:float,schedule:str='round_robin'):
        self.r_low = r_low
        self.r_high = r_high
        self.schedule = schedule
        self.selected = []
        self.unselected = []
        self.unique = []
        self.capacity = []
        self.jobs = {}

    def _update(self, selected_cids:List[str], unselected_cids:List[str], capcity:List[float], stragglers:Set):
        self.selected = selected_cids.copy()
        self.unselected = [cid for cid in unselected_cids if cid not in stragglers]
        self.capacity = capcity
        self.jobs={}

    def cluster(self, x:List[Union[float,int]]):
        y=sorted(x)
        th_high = y[int(self.r_high * len(y))]
        th_low  = y[int(self.r_low  * len(y))]
        print(f"th_high:{th_high} and th_low:{th_low}")

        for idx, val in enumerate(x):
            if val>= th_high:
                self.tier_high_cid.append((idx,val))
            elif val >= th_low:
                self.tier_mid_cid.append((idx,val))
            else:
                self.tier_low_cid.append((idx,val))

    def get_mappings(self,
                     selected_cids:List[str],
                     unselected_cids:List[str],
                     capacity:Dict[str, float],
                     stragglers:Set[str],
                     priority_sort=True
                     )-> Tuple[Dict[str, int], Dict[str,str], List[str]]:

        self._update(selected_cids, unselected_cids, capacity, stragglers,)

        if priority_sort:
            self.selected.sort(key= lambda x: capacity[x], reverse = False)
            self.unselected.sort(key= lambda x: capacity[x], reverse = True)

        if self.schedule == "round_robin":
            jobs={}
            mappings={}
            lim = len(self.unselected)

            for idx, cid in enumerate(self.selected):
                if cid in stragglers:
                    mappings[cid] = self.unselected_cids[idx%lim]
                    if self.unselected_cids[idx%lim] not in jobs:
                        jobs[self.unselected[idx % lim]] = 1
                    else:
                        jobs[self.unselected[idx % lim]]+=1

            return jobs, mappings,[*self.selected, *jobs.keys()]

        else: return {}, {}, []


if __name__=='__main__':
    flops = [15*n - n**2 for n in range(15)]
    c = Clustering_Module(flops,0.25, 0.75)
    c.get_tiers()
    c.print()
