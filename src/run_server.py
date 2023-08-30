import argparse
import os
import sys
from signal import SIGUSR1
import torch
import flwr
import logging

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#----Insert main project directory so that we can resolve the src imports-------
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, src_path)

from src import DEFAULT_SERVER_ADDRESS
from src.strategy import FedProxOffload
from src.ClientManager import OffloadClientManager
from src.Delegated import PPO
from src.utils import set_random_seed, BasicAgent

NUM_LAYERS_MODEL= 10
CHOP = 5
offload = True
def main()->None:
    parser = argparse.ArgumentParser(
        description='Flower Client instantiation.'
    )

    parser.add_argument("--sim",
                        required=False,
                        type=str,
                        help="Provide name of expriment to be performed.")
    parser.add_argument("--offload",
                        required=False,
                        type=bool,
                        help="Use offloading between clients"
                        )
    parser.add_argument("--address",
                        required=False,
                        type=str,
                        default=DEFAULT_SERVER_ADDRESS,
                        help="gRPC+socket client ip address")
    parser.add_argument("--idx",
                        required=False,
                        type=str,
                        help="Host number of socket")
    parser.add_argument("--cuda",
                        required=False,
                        type=bool,
                        default=False,
                        help="Enable GPU acceleration.")
    parser.add_argument("--seed",
                        required=False,
                        type=int,
                        default=0,
                        help="Seed to be used for reproducibility.")
    args = parser.parse_args()
    #------------------Server initialization------------------------
    set_random_seed(args.seed)

    client_manager = OffloadClientManager()

    # Initialize trained RL agent
    if offload:
        agent = PPO.PPO(state_dim=None,
                        action_dim=None,
                        action_std=None,
                        lr=None,
                        betas=None,
                        gamma=None,
                        K_epochs=None,
                        eps_clip=None,
                        )
        try:
            agent.policy.load_state_dict(torch.load('./PPO_FedAdapt.pth'))
        except:
            raise RuntimeError("Could not load RL agent")

    else:
        agent = BasicAgent(n1=CHOP)

    strategy = FedProxOffload(
        model_num_layers=NUM_LAYERS_MODEL,
        fraction_fit=None,
        fraction_evaluate=None,
        min_evaluate_clients=None,
        min_available_clients=None,
        evaluate_fn=None,
        on_fit_config_fn = None,
        on_evaluate_config_fn = None,
        accept_failures = True,
        initial_parameters = None,
        min_fit_clients=None,
        fit_metrics_aggregation_fn = None,
        evaluate_metrics_aggregation_fn = None,
        proximal_mu = 0.,
        agent = agent
    )

    #-------Flower server for federated learning------------------
    server = flwr.server.Server(
        client_manager = client_manager,
        strategy = strategy,
    )

    os.kill(os.getppid(), SIGUSR1)
    flwr.server.start_server(
        server_address=args.server_address,
        server=server,
    )

if __name__=='__main__':
    main()
