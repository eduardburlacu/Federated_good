import argparse
import os
import sys
#----Insert main project directory so that we can resolve the src imports-------
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, src_path)

from src import DEFAULT_SERVER_ADDRESS
from src.utils import set_random_seed

def main()->None:
    parser = argparse.ArgumentParser(
        description='Flower Client instantiation.'
    )
    parser.add_argument("--cid", required=True ,type=str, help="Provide cid of user.")
    parser.add_argument("--sim", required=True, type=str, help="Provide name of expriment to be performed.")
    parser.add_argument("--address", required=False,type=str, default=DEFAULT_SERVER_ADDRESS, help="gRPC+socket client address")
    parser.add_argument("--idx", required=True, type=str, help="Host index of socket")
    parser.add_argument("--seed", required=False,type=int, default=0, help="Seed to be used for reproducibility.")
    args = parser.parse_args()
    set_random_seed(args.seed)


if __name__=='__main__':
    main()
