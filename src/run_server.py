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
from src.strategy import FedProx_offload
from src.ClientManager import OffloadClientManager
from src.Delegated import PPO
from src.utils import set_random_seed


def main()->None:
    parser = argparse.ArgumentParser(
        description='Flower Client instantiation.'
    )
    offload = True
    parser.add_argument("--sim", required=True, type=str, help="Provide name of expriment to be performed.")
    parser.add_argument("--address", required=False,type=str, default=DEFAULT_SERVER_ADDRESS, help="gRPC+socket client address")
    parser.add_argument("--idx", required=True, type=str, help="Host index of socket")
    parser.add_argument("--seed", required=False,type=int, default=0, help="Seed to be used for reproducibility.")
    args = parser.parse_args()
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
        try: agent.policy.load_state_dict(torch.load('./PPO_FedAdapt.pth'))
        except: raise RuntimeError('Could not load RL agent')

    else: agent = None

    strategy = FedProx_offload(10,agent=agent)

    server = flwr.server.Server(
        client_manager = client_manager,
        strategy = strategy,
    )

    os.kill(os.getppid(), SIGUSR1)
    # Run server
    flwr.server.start_server(
        server_address=args.server_address,
        server=server,
    )


if __name__=='__main__':
    main()
