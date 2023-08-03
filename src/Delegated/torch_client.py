#----------------------------External Imports----------------------------
from copy import deepcopy
import flwr as fl
from flwr.common.typing import Scalar
import numpy as np
import ray
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List
#----------------------------Internal Imports-----------------------------
from src.utils import train, test, get_params, set_params, set_random_seed
from src.dataset_utils import get_dataloader
from src.script.parse_config import get_variables
from src.federated_dataset import load_data
from src.Models import FedAvg
from src.visualizer import summarize_loss

def get_FlowerClient_class(model, VARIABLES:Dict):
    class FlowerClient(fl.client.NumPyClient):
        model_class = model
        train_val_split = VARIABLES['VAL_SPLIT']
        min_num_samples = VARIABLES['MIN_DATASET_SIZE']
        properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        def __init__(self, cid: str, fed_dir_data: str, ):
            self.cid = cid
            self.fed_dir = Path(fed_dir_data)
            self.net = None
            self.device = torch.device(f"cuda:{int(cid)%torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")

        def get_parameters(self, config): return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

        def fit(self, parameters, config):
            set_random_seed(VARIABLES['SEED'])
            if self.net is None: self.net = FlowerClient.model_class(self.cid)
            set_params(self.net, parameters)
            num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
            if VARIABLES['DATASET'].name.lower() == 'cifar10':
                trainloader = get_dataloader(
                    self.fed_dir,
                    self.cid,
                    is_train=True,
                    batch_size=VARIABLES['BATCH_SIZE'],
                    workers=num_workers,
                )
            else:
                trainset = load_data(
                    client_names=[config['client_name']],
                    train_test_split=FlowerClient.train_val_split,
                    dataset_name=VARIABLES['DATASET'].name.lower(),
                    type="train",
                    min_no_samples=FlowerClient.min_num_samples,
                    is_embedded=bool(int(config["is_embedded"])))
                trainloader = DataLoader(
                    trainset, batch_size=VARIABLES['BATCH_SIZE'], shuffle=True )
            self.net.to(self.device)
            train(self.net, trainloader, epochs=VARIABLES["EPOCHS"], device=self.device)
            return get_params(self.net), len(trainloader.dataset), {}

        def evaluate(self, parameters, config):
            set_params(self.net, parameters)
            num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
            if VARIABLES['DATASET'].name.lower() == 'cifar10':
                valloader = get_dataloader(
                    self.fed_dir, self.cid, is_train=False, batch_size=VARIABLES['BATCH_SIZE'], workers=num_workers)
            else:
                testset = load_data(
                    client_names=[config['client_name']],
                    train_test_split=FlowerClient.train_val_split,
                    dataset_name=VARIABLES['DATASET'].name.lower(),
                    type="test",
                    min_no_samples=FlowerClient.min_num_samples,
                    is_embedded=bool(int(config["is_embedded"])))
                valloader = DataLoader(
                    testset, batch_size=VARIABLES['BATCH_SIZE'], shuffle=False ) #

            self.net.to(self.device) #
            loss, accuracy = test(self.net, valloader, device=self.device) #

            return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)} #

    return FlowerClient

def get_FlwrClient_class(model, VARIABLES:Dict):
    class FlwrClient(fl.client.NumPyClient):

        model_class = model
        train_test_split = VARIABLES['VAL_SPLIT']
        batch_size = VARIABLES['BATCH_SIZE']
        min_num_samples = VARIABLES['MIN_DATASET_SIZE']
        dataset_name = VARIABLES['DATASET'].name.lower()
        is_embedded = VARIABLES['IS_EMBEDDED']
        properties: Dict[str, Scalar] = {"tensor_type": "torch.Tensor"}
        def __init__(self, cid:str, plot_detailed_training=True):
            self.cid = cid
            self.net = None
            self.device = torch.device( f"cuda:{int(cid) % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
            self.first_client = None
            self.plot_detailed_training = plot_detailed_training
        def get_parameters(self, config):
            return self.net.get_weights()
        def set_parameters(self, params):
            self.net.set_weights(params)
        def fit(self, parameters, config):
            if self.net is None: self.net = FlwrClient.model_class()
            if self.first_client is None: self.first_client = config['first_client']
            set_random_seed(VARIABLES['SEED'])
            self.net.set_weights(parameters)
            optimizer = torch.optim.SGD(self.net.parameters(), lr= config['learning rate'])
            trainset = load_data(
                client_names=[config['client_name']],
                train_test_split=self.train_test_split,
                dataset_name= FlwrClient.dataset_name,
                type="train",
                min_no_samples = FlwrClient.min_num_samples,
                is_embedded = FlwrClient.is_embedded
            )
            trainloader = DataLoader(trainset, FlwrClient.batch_size, shuffle=True)
            if int(config['steps_per_epoch']) > 0:
                n_steps = min(int(config['steps_per_epoch']), len(trainloader))
            else:
                n_steps = len(trainloader)
            prev_global_params = deepcopy(list(self.net.parameters()))
            prev_global_params = [p.to(self.device) for p in prev_global_params ]
            self.net.train()
            self.net.to(self.device)
            for e in range(config['epochs']):
                for local_step, data in trainloader:
                    if local_step == n_steps: break
                    optimizer.zero_grad()
                    loss = self.net.train_step(data=data,
                                        mu = float(config["mu"]),
                                        old_params = prev_global_params )
                    loss.backward()
                    optimizer.step()
                    step = (config['epoch_global'] - 1) * config['epochs'] * n_steps + config['epoch'] * n_steps + local_step
                    if config['client_name'] == self.first_client and self.plot_detailed_training:
                        summarize_loss(f"training-detailed/{config['client_name']}", loss, step)
                    self.net.to(torch.device("cpu"))
            out_config ={} #### TO BE FILLED IN WHEN DOING STRATEGY
            return self.net.get_weights(), len(trainset), out_config

        def evaluate( self, parameters, config: Dict[str, Scalar]):
            raise EnvironmentError('Client evaluation called when we expected server side evaluation.')

    return FlwrClient

if __name__=='__main__':
    from src import GOD_CLIENT_NAME
    model = FedAvg.CNN_CIFAR
    CONFIG= get_variables('mock')
    FlwrClient = get_FlwrClient_class(model=model,VARIABLES=CONFIG)
    client = FlwrClient(GOD_CLIENT_NAME)