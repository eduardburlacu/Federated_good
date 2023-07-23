#----------------------------External Imports----------------------------
import argparse
import os
import sys
import random
import flwr as fl
from flwr.common.typing import Scalar
import ray
import torch
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

def set_random_seed(seed: int):
    random.seed(1+seed)
    np.random.seed(12 + seed)
    torch.manual_seed(123 + seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123 + seed) # Set seed for CUDA if available

#----Insert main project directory so that we can resolve the src imports-------

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, src_path)

#----------------------------Internal Imports-----------------------------
from src.utils import train, test, get_params, set_params, importer
from src.script.parse_config import get_variables
from src.dataset_utils import get_cifar_10, do_fl_partitioning, get_dataloader
from src.client import get_FlowerClient_class



