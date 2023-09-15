from typing import Tuple, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray


def capacity_distribution(
        num_clients:int,
):
    return {
        str(cid):1
        for cid in range(num_clients)
    }


def get_straggler_schedule(
        num_clients:int,
        num_rounds:int,
        stragglers_frac:float,
        type:str = "constant",
)-> NDArray:
    if type == "bernoulli":
        return np.transpose(
            np.random.choice(
                [0, 1],
                size = (num_rounds, num_clients),
                p = [1 - stragglers_frac, stragglers_frac],
            )
        )

    elif type == "constant":
        schedule = np.zeros(shape=(num_clients, num_rounds))
        target_clients = np.random.choice(
            [0,1],
            size = num_clients,
            p = [1 - stragglers_frac, stragglers_frac],
        )
        #flops_thresholds = [1.72, 0.315, 26, 0.727, 0.465, 0.158]
        computation_frac = {}
        for idx in range(num_clients):
            if target_clients[idx] == 1:
                schedule[idx,:] = np.ones(num_rounds)
                computation_frac[str(idx)] = np.random.beta(2,5)
            else:
                computation_frac[str(idx)] = 1.0

        return schedule, computation_frac

"""
def get_computation_frac( num_clients:int, beta=None )->NDArray:
    #useless
    if beta:
        return np.random.beta(beta,beta,size=num_clients)
    else:
        flops = np.array([1.72, 0.315, 26, 0.727, 0.465, 0.158])  # flops values in TFLOPS
        flops_probs = np.array([1 / 9, 1/9, 1/9, 1/3, 1/6, 1/6])
        sample_space = flops / flops.max() # normalize those values
        computation_frac = np.random.choice(
            sample_space,
            size=num_clients,
            p=flops_probs,
        )
    return computation_frac
"""
