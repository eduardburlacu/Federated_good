import argparse
import os
import sys
import flwr
import torch

#----Insert main project directory so that we can resolve the src imports----
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, src_path)

from src import DEFAULT_SERVER_ADDRESS
from src.utils import set_random_seed, server_address

def main()->None:
    #--------------Setup inputs to client------------------------------------
    parser = argparse.ArgumentParser(
        description='Flower Client instantiation.'
    )
    parser.add_argument("--cid",
                        required=True ,
                        type=str,
                        help="Provide cid of user.")
    parser.add_argument("--sim",
                        required=True,
                        type=str,
                        help="Provide name of expriment to be performed.")
    parser.add_argument("--address",
                        required=False,
                        type=str,
                        default=DEFAULT_SERVER_ADDRESS,
                        help="gRPC+socket client ip address")
    parser.add_argument("--idx",
                        required=True,
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
    #------------------ Client initialization ---------------------------
    set_random_seed(args.seed)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.use_cuda
        else "cpu"
    )
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    client=None

    #--------------------------Start Client------------------------------
    flwr.client.start_numpy_client(
        server_address=server_address(
            args.address,
            args.idx
        ),
        client=client
    )



if __name__=='__main__':
    main()
