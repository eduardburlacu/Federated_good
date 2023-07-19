#----------------------------External Imports----------------------------
import argparse
import flwr as fl
from flwr.common.typing import Scalar
import ray
import torch
import torchvision
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple, List

#----Insert main project directory so that we can resolve the src imports-------
import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, src_path)

#----------------------------Internal Imports-----------------------------
from src.utils import Net, train, test, get_params, set_params, importer
from src.script.parse_config import get_variables
from src.dataset_utils import get_cifar_10, do_fl_partitioning, get_dataloader

#-------------------------------Setup parser------------------------------
parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
parser.add_argument('--config', type=str, default='mock')
parser.add_argument('--size', type=str, default='small')

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, fed_dir_data: str):
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}

        # Instantiate model
        self.net = Net()

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config):
        return get_params(self.net)

    def fit(self, parameters, config):
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        trainloader = get_dataloader(
            self.fed_dir,
            self.cid,
            is_train=True,
            batch_size=config["batch_size"],
            workers=num_workers,
        )

        # Send model to device
        self.net.to(self.device)

        # Train
        train(self.net, trainloader, epochs=config["epochs"], device=self.device)

        # Return local model and statistics
        return get_params(self.net), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        valloader = get_dataloader(
            self.fed_dir, self.cid, is_train=False, batch_size=50, workers=num_workers
        )

        # Send model to device
        self.net.to(self.device)

        # Evaluate
        loss, accuracy = test(self.net, valloader, device=self.device)

        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def get_evaluate_fn( testset: torchvision.datasets.CIFAR10, ) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = Net()
        set_params(model, parameters)
        model.to(device)

        testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        loss, accuracy = test(model, testloader, device=device)

        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate

# -------------------------------Example----------------------------------------------
# 1. Downloads CIFAR-10
# 2. Partitions the dataset into N splits, where N is the total number of
#    clients. We refere to this as `pool_size`. The partition can be IID or non-IID
# 3. Starts a simulation where a % of clients are sample each round.
# 4. After the M rounds end, the global model is evaluated on the entire testset.
#    Also, the global model is evaluated on the valset partition residing in each client.
if __name__ == "__main__":

    args = parser.parse_args()                         # parse input arguments
    CONFIG = get_variables(args.config)                # Get toml config
    client_resources = { "num_cpus": CONFIG['CPUS'] }  # each client will get allocated this many CPUs in simulation.
    models = importer(CONFIG['STRATEGY'])              #src/Models/Relevant file aliased as models
    train_path, testset = get_cifar_10()
    fed_dir = do_fl_partitioning(                                                # use a large `alpha` to make it IID;
        train_path, pool_size=CONFIG['NUM_CLIENTS'], alpha=CONFIG['ALFA'],       # a small value (e.g. 1) will make it non-IID
        num_classes=CONFIG['DATASET'].num_classes, val_ratio=CONFIG['VAL_SPLIT'] # This will create a new directory called "federated": in the directory where
    )                                                                            # CIFAR-10 lives. Inside it, there will be N=NUM_CLIENTS sub-directories
                                                                                 # each with its own train/set split.
    # ------------------------------------Simulation------------------------------------
    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "epochs": CONFIG['EPOCHS'],  # number of local epochs
            "batch_size": ['BATCH_SIZE'],
        }
        return config

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=CONFIG['FRAC_FIT'],
        fraction_evaluate=CONFIG['FRAC_EVALUATE'],
        min_fit_clients=CONFIG['MIN_FIT_CLIENTS'],
        min_evaluate_clients=CONFIG['MIN_EVALUABLE_CLIENTS'],
        min_available_clients=CONFIG['MIN_AVAILABLE_CLIENTS'], # All clients should be available
        on_fit_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn(testset),  # centralised evaluation of global model
    )

    def client_fn(cid: str):
        # create a single client instance
        return FlowerClient(cid, fed_dir)

    ray_init_args = {"include_dashboard": False} # (optional) specify Ray config

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=CONFIG['NUM_CLIENTS'],
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=CONFIG['NUM_ROUNDS']),
        strategy=strategy,
        ray_init_args=ray_init_args,
    )
