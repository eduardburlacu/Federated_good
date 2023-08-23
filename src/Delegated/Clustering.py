from typing import List

class Clustering_Module:
    def __init__(self,
                 flops: List[int],
                 r_low:float,
                 r_high:float,
                 ):
        self.flops = flops
        self.client_number = len(flops)
        self.r_low = r_low
        self.r_high = r_high

        self.tier_high_cid= []
        self.tier_mid_cid = []
        self.tier_low_cid = []


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
            self.importance_sort()

    def importance_sort(self):
        self.tier_high_cid.sort(key= lambda x:x[1], reverse=True)
        self.tier_low_cid.sort(key=lambda x: x[1], reverse=False)

    def print(self):
        print("Tier high queue-->",self.tier_high_cid)
        print("Tier mid queue-->" ,self.tier_mid_cid)
        print("Tier low queue-->" ,self.tier_low_cid)



if __name__=='__main__':
    flops = [15*n - n**2 for n in range(15)]
    c = Clustering_Module(flops,0.25, 0.75)
    c.get_tiers()
    c.print()
