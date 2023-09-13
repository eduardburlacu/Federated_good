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
        type:str = "bernoulli",
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
        for idx in range(num_clients):
            if target_clients[idx] == 1:
                schedule[idx,:] = np.ones(num_rounds)

        return schedule

