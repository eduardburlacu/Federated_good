import argparse
import os
import sys
from signal import SIGUSR1
import torch
import flwr
from flwr.server import ServerConfig
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
import logging

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#----Insert main project directory so that we can resolve the src imports-------
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, src_path)

from src import DEFAULT_GRPC_ADDRESS, PATH_src
from src.server import gen_evaluate_fn
from src.ClientManager import OffloadClientManager
from src.Dataset.dataset import load_datasets
from src.utils import set_random_seed, get_ports, save_results_as_pickle, plot_metric_from_history

@hydra.main(config_path=PATH_src["conf"], config_name="config_offload", version_base=None)
def main(cfg: DictConfig)->None:
    parser = argparse.ArgumentParser(
        description='Flower Server instantiation.'
    )

    parser.add_argument("--sim",
                        required=False,
                        type=int,
                        default=1,
                        help="Provide name of expriment to be performed.")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_GRPC_ADDRESS,
        help="gRPC+Socket server address)",
    )

    parser.add_argument("--seed",
                        required=False,
                        type=int,
                        default=0,
                        help="Seed to be used for reproducibility.")
    args = parser.parse_args()

    #------------------Server initialization------------------------
    set_random_seed(args.seed)
    ports = get_ports(cfg.num_clients)
    client_manager = OffloadClientManager()
    agent = instantiate(cfg.agent)
    if repr(agent)=="PPO":
        try:
            agent.policy.load_state_dict(torch.load('./PPO_FedAdapt.pth'))
        except:
            raise RuntimeError("Could not load RL agent")
    device = cfg.server_device
    testloader, datasizes = load_datasets(
        config=cfg.dataset_config,
        num_clients=cfg.num_clients,
        batch_size=cfg.batch_size,
    )[-2:]
    evaluate_fn = gen_evaluate_fn(testloader, device=device, model=cfg.model)
    init_stragglers = {str(cid): 0 for cid in range(cfg.num_clients)}
    base_capacity = 1 / cfg.num_clients
    init_capacities = {str(cid): base_capacity for cid in range(cfg.num_clients)}
    del base_capacity

    if repr(agent)=="PPO":
        try:
            agent.policy.load_state_dict(torch.load('./PPO_FedAdapt.pth'))
        except:
            raise RuntimeError("Could not load RL agent")


    strategy = instantiate(
        cfg.strategy,
        evaluate_fn=evaluate_fn,
        agent=agent,
        init_stragglers=init_stragglers,
        init_capacities=init_capacities,
        ports=ports,
    )

    #-------Flower server for federated learning------------------
    server = flwr.server.Server(
        client_manager = client_manager,
        strategy = strategy,
    )

    history = flwr.server.start_server(
        server_address=args.server_address,
        server=server,
        config=ServerConfig(cfg.num_rounds),
    )

    # Experiment completed. Now we save the results and
    # generate plots using the `history`
    print("................")
    print(history)

    # Hydra automatically creates an output directory
    # Let's retrieve it and save some results there
    save_path = HydraConfig.get().runtime.output_dir

    # save results as a Python pickle using a file_path
    # the directory created by Hydra for each run
    save_results_as_pickle(history, file_path=save_path, extra_results={})

    # plot results and include them in the readme
    strategy_name = strategy.__class__.__name__
    file_suffix: str = (
        f"_{strategy_name}"
        f"{'_iid' if cfg.dataset_config.iid else ''}"
        f"{'_balanced' if cfg.dataset_config.balance else ''}"
        f"{'_powerlaw' if cfg.dataset_config.power_law else ''}"
        f"_C={cfg.num_clients}"
        f"_B={cfg.batch_size}"
        f"_E={cfg.num_epochs}"
        f"_R={cfg.num_rounds}"
        f"_mu={cfg.mu}"
        f"_strag={cfg.stragglers_fraction}"
    )

    plot_metric_from_history(
        history,
        save_path,
        (file_suffix),
    )
    os.kill(os.getppid(), SIGUSR1)

if __name__=='__main__':
    main()
