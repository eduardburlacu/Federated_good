#----------------------------External Imports----------------------------
import os
import flwr as fl
from flwr.common.typing import Scalar
import ray
import torch
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Dict
#----------------------------Internal Imports-----------------------------
from src.utils import Net, train, test, get_params, set_params, importer, set_random_seed
from src.dataset_utils import get_dataloader
from src.script.parse_config import get_variables
from src.federated_dataset import load_data
from src.Models import FedAvg

def get_FlowerClient_class(model, CONFIG:Dict):
    class FlowerClient(fl.client.NumPyClient):
        model_class = model
        train_val_split = CONFIG['VAL_SPLIT']
        min_num_samples = CONFIG['MIN_DATASET_SIZE']
        properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        def __init__(self, cid: str, fed_dir_data: str, ):
            self.cid = cid
            self.fed_dir = Path(fed_dir_data)
            self.net = None
            self.device = torch.device(f"cuda:{int(cid)%torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")

        def get_parameters(self, config): return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

        def fit(self, parameters, config):
            set_random_seed(CONFIG['SEED'])
            if self.net is None: self.net = FlowerClient.model_class(self.cid)
            set_params(self.net, parameters)
            num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
            if CONFIG['DATASET'].name.lower() == 'cifar10':
                trainloader = get_dataloader(
                    self.fed_dir,
                    self.cid,
                    is_train=True,
                    batch_size=CONFIG['BATCH_SIZE'],
                    workers=num_workers,
                )
            else:
                trainset = load_data(
                    client_names=[config['client_name']],
                    train_test_split=FlowerClient.train_val_split,
                    dataset_name=CONFIG['DATASET'].name.lower(),
                    type="train",
                    min_no_samples=FlowerClient.min_num_samples,
                    is_embedded=bool(int(config["is_embedded"])))
                trainloader = DataLoader(
                    trainset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True )
            self.net.to(self.device)
            train(self.net, trainloader, epochs=CONFIG["EPOCHS"], device=self.device)
            return get_params(self.net), len(trainloader.dataset), {}

        def evaluate(self, parameters, config):
            set_params(self.net, parameters)
            num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
            if CONFIG['DATASET'].name.lower() == 'cifar10':
                valloader = get_dataloader(
                    self.fed_dir, self.cid, is_train=False, batch_size=CONFIG['BATCH_SIZE'], workers=num_workers)
            else:
                testset = load_data(
                    client_names=[config['client_name']],
                    train_test_split=FlowerClient.train_val_split,
                    dataset_name=CONFIG['DATASET'].name.lower(),
                    type="test",
                    min_no_samples=FlowerClient.min_num_samples,
                    is_embedded=bool(int(config["is_embedded"])))
                valloader = DataLoader(
                    testset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False ) #

            self.net.to(self.device) #
            loss, accuracy = test(self.net, valloader, device=self.device) #

            return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)} #

    return FlowerClient

if __name__=='__main__':
    model = FedAvg.CNN_CIFAR
    CONFIG= get_variables('mock')
    get_FlowerClient_class(model=model,CONFIG=CONFIG)