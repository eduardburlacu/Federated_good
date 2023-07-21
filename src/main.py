#----------------------------External Imports----------------------------
import argparse
import flwr as fl
from flwr.common.typing import Scalar
import ray
import torch
import torchvision
import numpy as np
from collections import OrderedDict
from typing import Dict, Callable, Optional, Tuple, List

#----Insert main project directory so that we can resolve the src imports-------
import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, src_path)

#----------------------------Internal Imports-----------------------------
from src.utils import train, test, get_params, set_params, importer
from src.script.parse_config import get_variables, prepare_data
from src.dataset_utils import get_cifar_10, do_fl_partitioning, get_dataloader
from src.client import FlowerClient

#-------------------------------Setup parser------------------------------
parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
parser.add_argument('--config', type=str, default='mock')
parser.add_argument('--size', type=str, default='small')

# -------------------------------Example----------------------------------------------
# 1. Downloads CIFAR-10
# 2. Partitions the dataset into N splits, where N is the total number of
#    clients. We refere to this as `pool_size`. The partition can be IID or non-IID
# 3. Starts a simulation where a % of clients are sample each round.
# 4. After the M rounds end, the global model is evaluated on the entire testset.
#    Also, the global model is evaluated on the valset partition residing in each client.

if __name__ == "__main__":
    #-----------------------------------Pipeline configuration, datasets, models------------------------------------
    args = parser.parse_args()                         # parse input arguments
    CONFIG = get_variables(args.config)                # Get toml config
    client_resources = { "num_cpus": CONFIG['CPUS'] }  # each client will get allocated this many CPUs in simulation.
    module = importer(CONFIG['AGGREGATOR'])            # src/Models/Relevant file aliased as module
    model, datanode = module.load_models_datanodes(CONFIG['MODEL'],CONFIG['DATASET'],CONFIG['IID'])   # Draw from available
    # -----------------------------------------------Simulation-----------------------------------------------------
    def get_evaluate_fn(testset: torchvision.datasets.CIFAR10, ) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
        """Return an evaluation function for centralized evaluation."""

        def evaluate( server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar] ) -> Optional[Tuple[float, float]]:
            """Use the entire CIFAR-10 test set for evaluation."""
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # determine device
            net = model()
            set_params(net, parameters)
            net.to(device)
            testloader = torch.utils.data.DataLoader(testset, batch_size=50)
            loss, accuracy = test(net, testloader, device=device)
            return loss, {"accuracy": accuracy} # return metrics

        return evaluate

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "epochs": CONFIG['EPOCHS'],  # number of local epochs
            "batch_size": CONFIG['BATCH_SIZE'],
        }
        return config

    #--------------------------------------------Data preparation---------------------------------------------------
    prepare_data(datanode)
    train_path, testset = get_cifar_10()
    fed_dir = do_fl_partitioning(
        train_path,
        pool_size=CONFIG['NUM_CLIENTS'],            # use a large `alpha` to make it IID; a small value (e.g. 1) will make it non-IID
        alpha=CONFIG['ALFA'],                       # This will create a new directory called "federated": in the directory where CIFAR-10 lives.
        num_classes=CONFIG['DATASET'].num_classes,  # Inside it, there will be N=NUM_CLIENTS sub-directories each with its own train/set split.
        val_ratio=CONFIG['VAL_SPLIT']
    )

    #------------------------------------------------Strategy-------------------------------------------------------
    def client_fn(cid: str):
        # create a single client instance
        return FlowerClient(cid=cid, fed_dir_data=fed_dir, model_class=model)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=CONFIG['FRAC_FIT'],
        fraction_evaluate=CONFIG['FRAC_EVALUATE'],
        min_fit_clients=CONFIG['MIN_FIT_CLIENTS'],
        min_evaluate_clients=CONFIG['MIN_EVALUABLE_CLIENTS'],
        min_available_clients=CONFIG['MIN_AVAILABLE_CLIENTS'],
        on_fit_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn(testset),  # centralised evaluation of global model
    )

    ray_init_args = {"include_dashboard": False} # (optional) specify Ray config

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=CONFIG['NUM_CLIENTS'],
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=CONFIG['NUM_ROUNDS']),
        strategy=strategy,
        ray_init_args=ray_init_args,
    )
