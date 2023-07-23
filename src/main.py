#----------------------------External Imports----------------------------
import argparse
import os
import sys
import random
import flwr as fl
from flwr.common.typing import Scalar
import ray
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
from collections import OrderedDict
from typing import Dict, Callable, Optional, Tuple, List

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
    )
else: DEVICE = torch.device('cpu')

#----Insert main project directory so that we can resolve the src imports-------
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, src_path)

#----------------------------Internal Imports-----------------------------
from src.utils import train, test, get_params, set_params, importer, set_random_seed
from src.script.parse_config import get_variables
from src.dataset_utils import get_cifar_10, do_fl_partitioning, get_dataloader
from src.client import get_FlowerClient_class
from src import PATH, GOD_CLIENT_NAME
#-------------------------------Setup parser------------------------------
parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
parser.add_argument('--config',
                    help='name of configuration file (without .toml)',
                    type=str,
                    default='mock')

if __name__ == "__main__":
    #-----------------------------------Pipeline configuration, datasets, models-------------------
    args = parser.parse_args()                         # parse input arguments
    CONFIG = get_variables(args.config)                # Get toml config
    set_random_seed(CONFIG['SEED'])
    if DEVICE.type=='cuda':
        client_resources = {'num_gpus': torch.cuda.device_count()}
    else: client_resources = { "num_cpus": CONFIG['CPUS'] }
    module = importer(CONFIG['AGGREGATOR'])      # src/Models/Relevant file aliased as module
    model, datanode = module.load_models_datanodes(CONFIG['MODEL'],CONFIG['DATASET'].name,CONFIG['IID'])
    FlowerClient = get_FlowerClient_class(model, CONFIG)
    writer = SummaryWriter(PATH['logs'])
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
            writer.add_scalar("Loss/train", loss, server_round)
            writer.add_scalar("Accuracy/train", accuracy, server_round)
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

    if CONFIG['DATASET'].lower() in {'cifar10'}:
        train_path, testset = get_cifar_10()
        fed_dir = do_fl_partitioning(
            train_path,
            pool_size=CONFIG['NUM_CLIENTS'],            # use a large `alpha` to make it IID; a small value (e.g. 1) will make it non-IID
            alpha=CONFIG['ALFA'],                       # This will create a new directory called "federated": in the directory where CIFAR-10 lives.
            num_classes=CONFIG['DATASET'].num_classes,  # Inside it, there will be N=NUM_CLIENTS sub-directories each with its own train/set split.
            val_ratio=CONFIG['VAL_SPLIT']
        )

    '''
    The following facts should be mirrored: 
    - I need a list with all clients and their sizes ---> json/csv file
    - I need to make all CONFIG files for FedProx paper  
    - Validate everything before simulation
    - Simulation
    - Improve efficiency by Lorenzo insight
    '''
    #------------------------------------------------Strategy-------------------------------------------------------
    def client_fn(cid: str): return FlowerClient(cid=cid, fed_dir_data=fed_dir, model_class=model)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=CONFIG['FRAC_FIT'],                        # Sample _% of available clients for training round
        fraction_evaluate=CONFIG['FRAC_EVALUATE'],              # Sample _% of available clients for evaluation round
        min_fit_clients=CONFIG['MIN_FIT_CLIENTS'],              # Never sample less than _ clients for training
        min_evaluate_clients=CONFIG['MIN_EVALUABLE_CLIENTS'],   # Never sample less than _ clients for evaluation
        min_available_clients=CONFIG['MIN_AVAILABLE_CLIENTS'],  # Wait until _ clients are available
        on_fit_config_fn=fit_config,                            #
        evaluate_fn=get_evaluate_fn(testset),                   # centralised eval of global model

    )

    ray_init_args = {"include_dashboard": False}                # (optional) specify Ray config

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=CONFIG['NUM_CLIENTS'],
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=CONFIG['NUM_ROUNDS']),
        strategy=strategy,
        ray_init_args=ray_init_args,
    )
    writer.close()
