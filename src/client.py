#----------------------------External Imports----------------------------
import os
import flwr as fl
from flwr.common.typing import Scalar
import ray
import torch
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Dict
#----------------------------Internal Imports-----------------------------
from src.utils import Net, train, test, get_params, set_params, importer
from src.dataset_utils import get_cifar_10, do_fl_partitioning, get_dataloader

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

        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

        def fit(self, parameters, config):
            if self.net is None: self.net = FlowerClient.model_class(self.cid)
            set_params(self.net, parameters)
            num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
            trainloader = get_dataloader(
                self.fed_dir,
                self.cid,
                is_train=True,
                batch_size=CONFIG['BATCH_SIZE'],
                workers=num_workers,
            )
            self.net.to(self.device)
            train(self.net, trainloader, epochs=CONFIG["EPOCHS"], device=self.device)
            return get_params(self.net), len(trainloader.dataset), {}

        def evaluate(self, parameters, config):
            set_params(self.net, parameters)
            num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
            valloader = get_dataloader(
                self.fed_dir, self.cid, is_train=False, batch_size=CONFIG['BATCH_SIZE'], workers=num_workers
            )

            self.net.to(self.device)
            loss, accuracy = test(self.net, valloader, device=self.device)

            return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}

    return FlowerClient