import argparse
import os
import sys
import signal
import subprocess
import flwr
import torch

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

#----Insert main project directory so that we can resolve the src imports----
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, src_path)

from src import DEFAULT_GRPC_ADDRESS, DEFAULT_SERVER_ADDRESS, PATH_src
from src.client import gen_client_fn
from src.Dataset.dataset import load_datasets
from src.utils import set_random_seed, get_ports

process = subprocess.Popen(["python","-m","src."])
child_processes = []

def on_server_ready(signum):
    assert signum == signal.SIGUSR1


@hydra.main(config_path=PATH_src["conf"], config_name="config_offload", version_base=None)
def main(cfg: DictConfig)->None:
    #--------------Setup inputs to client------------------------------------
    parser = argparse.ArgumentParser(
        description='Flower Client instantiation.'
    )

    parser.add_argument("--seed",
                        required=False,
                        type=int,
                        default=0,
                        help="Seed to be used for reproducibility.")
    args = parser.parse_args()

    if cfg.dataset.lower() in {"mnist","cifar10"}:
        trainloaders, valloaders, testloader, datasizes = load_datasets(
            config=cfg.dataset_config,
            num_clients=cfg.num_clients,
            batch_size=cfg.batch_size,
        )
        ports = get_ports(cfg.num_clients)

        init_stragglers = {str(cid): 0 for cid in range(cfg.num_clients)}
        base_capacity = 1 / cfg.num_clients
        init_capacities = {str(cid): base_capacity for cid in range(cfg.num_clients)}
        del base_capacity

        client_fn = gen_client_fn(
            num_clients=cfg.num_clients,
            num_rounds=cfg.num_rounds,
            num_epochs=cfg.num_epochs,
            trainloaders=trainloaders,
            valloaders=valloaders,
            learning_rate=cfg.learning_rate,
            stragglers_frac=cfg.stragglers_fraction,
            capacities=init_capacities,
            model=cfg.model,
            ip_address=DEFAULT_SERVER_ADDRESS,
            ports=ports,
        )
    else:
        raise AttributeError("Dataset name not supported yet.")


    # --------------------------Launch Clients----------------------------
    for cid in range(cfg.num_clients):
        print(f"Starting CLIENT {cid}/{cfg.num_clients}...")
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and args.use_cuda
            else "cpu"
        )

        try:
            client = client_fn(str(cid))
            flwr.client.start_numpy_client(
                server_address=DEFAULT_GRPC_ADDRESS,
                client=client,
            )

        except Exception as e:
            raise e

if __name__=='__main__':
    main()
