def generate_clusters(num_clusters:int, num_clients:int,sampler):
    '''
    For each cid, it will return {cid:set(neighbours)}, to be passed to strategy.
    :param num_clusters: The number of clusters partitioned
    :param sampler: How to distribute data, use the various parameters of heterogeneity
    :return: se
    '''
    pass
def update_clusters():
    '''
    For each cluster, it updates the 3 metrics.
    :return:
    '''

    pass
def generate_flops(num_clients:int, sampler):
    '''

    :param num_clients: The number of clusters partitioned
    :param sampler: The sampling rule
    :return:
    '''

    pass

def update_capacity(client):
    '''
    Returns the current capacity score of th device
    :param client:
    :return:
    '''
    pass

def update_tier(capacities:list(int), treshold:float = 0.6):
    '''
    def update_tier(self, round :int,  total_transfer:float):
        if round==1 or self.transfer_rate is None:
            self.capacity = self.flops / self.total_flop_size
        elif self.transfer_rate==0. :
            self.capacity = (self.flops / self.total_flop_size) / ((self.dataset_size / self.transfer_rate) / total_transfer)
    :param capacities:
    :param treshold:
    :return:
    '''
    pass
def KL_div():
    pass
def get_cluster_metrics():
    pass


def to_probability(array): return array/sum(array)