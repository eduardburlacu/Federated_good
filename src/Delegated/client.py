"""Defines the MNIST Flower Client and a function to instantiate it."""

import time
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
#from pathlib import Path

#from src.dataset import load_datasets
from src.federated_dataset import load_data
from src.Delegated.models import test, train
from src.Delegated.Communication import Communicator
from src.Delegated.split_learn import split_model, send_msg, recv_msg


def timeit(func: Callable)-> Callable:
    def wrapper(self, *args, **kwargs):
        start = time.time()
        func(self, *args, **kwargs)
        end = time.time()
        self.time = end - start

    return wrapper

class FlowerClient(
    fl.client.NumPyClient,
    Communicator
):  # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        cid:str,
        model: DictConfig,
        trainloader: DataLoader,
        valloader: DataLoader,
        datasize: float,
        device: torch.device,
        num_epochs: int,
        flop_rate:int,
        learning_rate: float,
        straggler_schedule: np.ndarray,
        index:str,
        ip_address:str,
    ):  # pylint: disable=too-many-arguments
        self.cid = cid
        self.net = None
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs

        self.flop_rate = flop_rate
        self.datasize = datasize
        self.computation_frac:float = 1.0
        self.mbps = None
        self.time = None
        self.capacity = None

        self.learning_rate = learning_rate
        self.straggler_schedule = straggler_schedule
        Communicator.__init__(self, index, ip_address)
        # !!!!!!!!!!!! self.connect(server credentials)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Returns the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Changes the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def get_speed(self, model_size:float):
        if self.connected:
            network_time_start = time.time()
            msg = ['MSG_TEST_NETWORK', self.net.cpu().state_dict()]
            self.send_msg(
                self.sock,
                msg
            )
            msg = self.recv_msg(self.sock, 'MSG_TEST_NETWORK')[1]
            network_time_end = time.time()
            self.mbps = (2 * model_size * 8) / (network_time_end - network_time_start)  # Mbit/s
        else: #Use -1 to mark lack of connection
            self.mbps = -1.


    def split_train(
            self,
            split_layer:int,
            trainloader: DataLoader,
            device: torch.device,
            epochs: int,
            learning_rate: float,
            client_ip: str,
            frac: float = 1.0,
            momentum: float = 0.9
    )->None:

        self.net = split_model(self.net, split_layer)[0]
        global_params = [val.detach().clone() for val in self.net.parameters()]
        self.net.train()
        optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=learning_rate,
            momentum=momentum,
        )

        for e in range(epochs):
            for batch_idx, (features, targets) in enumerate(trainloader):
                if batch_idx<= int(frac* len(trainloader)):
                    features, targets = features.to(device), targets.to(device)
                    optimizer.zero_grad()
                    proximal_term = 0.0
                    for local_weights, global_weights in zip(self.net.parameters(), global_params):
                        proximal_term += (local_weights - global_weights).norm(2)
                    smashed_activations = self.net(features)
                    # Transfer data to other device
                    msg = ['MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER',
                           split_layer,
                           proximal_term,
                           smashed_activations.cpu(),
                           targets.cpu()
                           ]
                    send_msg(
                        self.sock,
                        msg
                    )
                    # Wait for backprop
                    gradients = recv_msg(self.sock)[1].to(device)
                    smashed_activations.backward(gradients)

        msg = ['MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER',
               self.cid,
               self.get_parameters({}),
               len(self.trainloader)
               ]
        send_msg(self.sock,msg)

    def split_follower(
            self,
            device: torch.device,
            learning_rate: float,
            proximal_mu: float,
            momentum: float = 0.9
    ):
        # Wait for forward prop
        msg = recv_msg(self.sock, 'MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER')


        if len(msg)==4:
            cid, num_examples, ante_parameters = msg[1:]
            return [*ante_parameters, *self.get_parameters({})], num_examples, {"split":cid}

        else:
            criterion = nn.CrossEntropyLoss()

            split_layer, proximal_term, smashed_activations, targets = msg[1:]
            self.net = split_model(self.net, int(split_layer))[1]
            self.net.train()
            global_params = [val.detach().clone() for val in self.net.parameters()]
            optimizer = torch.optim.SGD(
                self.net.parameters(),
                lr=learning_rate,
                momentum=momentum,
            )
            smashed_activations, targets = smashed_activations.to(device), targets.to(device)
            optimizer.zero_grad()

            for local_weights, global_weights in zip(self.net.parameters(), global_params):
                proximal_term += (local_weights - global_weights).norm(2)
            output = self.net(smashed_activations)
            loss = criterion(output, targets) + proximal_mu * proximal_term /2
            loss.backward()
            optimizer.step()
            # Send gradients to client
            msg = ['MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT', smashed_activations.grad]
            send_msg(self.sock, msg)

            return None

    @timeit
    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implements distributed fit function for a given client.

        At each round check if the client is a straggler,
        if so, train less epochs (to simulate partial work)
        if the client is told to be dropped (e.g. because not using
        FedProx in the server), the fit method returns without doing
        training.
        This method always returns via the metrics (last argument being
        returned) whether the client is a straggler or not. This info
        is used by strategies other than FedProx to discard the update.
        """

        if self.net is None:
            self.net = instantiate(self.model).to(self.device)

        self.set_parameters(parameters)

        if (
                self.straggler_schedule[int(config["curr_round"]) - 1]
                and self.num_epochs > 1
        ):
            num_epochs = np.random.randint(1, self.num_epochs)
            self.computation_frac = num_epochs / self.num_epochs

            if config["drop_client"]:
                # return without doing any training.
                # The flag in the metric will be used to tell the strategy
                # to discard the model upon aggregation
                return (
                    self.get_parameters({}),
                    len(self.trainloader),
                    {"is_straggler": True},
                )

        else:
            num_epochs = self.num_epochs
            self.computation_frac = 1.

        self.capacity = self.datasize * (1. - self.computation_frac)

        if config['follower']:
            #self.get_speed({})
            #self.connect()
            result = None
            while result is None:
                result= self.split_follower(
                    self.device,
                    self.learning_rate,
                    config["proximal_mu"]
                )
            return result

        else:

            if config["split_layer"] == len(list(self.net.children()))-1:  # No offloading training

                train(
                    self.net,
                    self.trainloader,
                    self.device,
                    epochs=num_epochs,
                    learning_rate=self.learning_rate,
                    proximal_mu=config["proximal_mu"],
                    frac= config["frac"]
                )
                return self.get_parameters({}), len(self.trainloader), {"is_straggler": False, "split": None, "next":self.straggler_schedule[int(config["curr_round"])]}

            else: # Offload a part of the training
                # self.get_speed({})
                # self.connect()
                if len(list(self.net.children())) != len(parameters):
                    self.net = instantiate(self.model).to(self.device)

                self.set_parameters(parameters)

                if not self.connected:
                    self.connect(config['other_index'],config['other_ip'])

                self.split_train(
                    split_layer=config["split_layer"],
                    trainloader=self.trainloader,
                    device=self.device,
                    epochs=num_epochs,
                    learning_rate=self.learning_rate,
                    client_ip = self.ip,
                    frac = 1.0,
                )
                #self.get_speed(config["model_size"])
                #self.disconnect()
                return [], len(self.trainloader), {"is_straggler": False, "split":self.cid, "next":self.straggler_schedule[int(config["curr_round"])] }

    @timeit
    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implements distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader, self.device, config["frac"])
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def gen_client_fn(
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    datasizes: List[float],
    learning_rate: float,
    stragglers: float,
    flop_rates: List[int],
    indexes: List[str],
    ips: List[str],
    model: DictConfig,
) -> Callable[[str], FlowerClient]:  # pylint: disable=too-many-arguments
    """Generates the client function that creates the Flower Clients.

    Parameters
    ----------
    num_clients : int
        The number of clients present in the setup
    num_rounds: int
        The number of rounds in the experiment. This is used to construct
        the scheduling for stragglers
    num_epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    trainloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset training partition
        belonging to a particular client.
    valloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset validation partition
        belonging to a particular client.
    learning_rate : float
        The learning rate for the SGD  optimizer of clients.
    stragglers : float
        Proportion of stragglers in the clients, between 0 and 1.
    ips: Dict[str,str]
    indexes: Dict[str,str]
    flop_rates: Dict[str,str]

    Returns
    -------
    Tuple[Callable[[str], FlowerClient], DataLoader]
        A tuple containing the client function that creates Flower Clients and
        the DataLoader that will be used for testing
    """

    # Defines a staggling schedule for each clients, i.e at which round will they
    # be a straggler. This is done so at each round the proportion of staggling
    # clients is respected
    stragglers_mat = np.transpose(
        np.random.choice(
            [0, 1], size=(num_rounds, num_clients), p=[1 - stragglers, stragglers]
        )
    )

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #Load data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        return FlowerClient(
            cid,
            model,
            trainloader,
            valloader,
            datasizes[int(cid)],
            device,
            num_epochs,
            flop_rates[int(cid)],
            learning_rate,
            stragglers_mat[int(cid)],
            indexes[int(cid)],
            ips[int(cid)],
        )

    return client_fn


class FedFlowerClient(
    FlowerClient
):  # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        net: torch.nn.Module,
        client_name:str,
        dataset_name:str,
        train_test_split: float,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        straggler_schedule: np.ndarray,
        batch_size:int =32,
        min_num_samples:int =10,
    ):  # pylint: disable=too-many-arguments
        self.net = net
        self.client_name = client_name
        self.dataset_name =dataset_name
        self.train_test_split = train_test_split
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.straggler_schedule = straggler_schedule
        self.batch_size = batch_size
        self.min_num_samples = min_num_samples
        self.trainloader = None
        self.testloader = None
        self.is_embedded: bool = self.dataset_name == "shakespeare"
    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implements distributed fit function for a given client.

        At each round check if the client is a straggler,
        if so, train less epochs (to simulate partial work)
        if the client is told to be dropped (e.g. because not using
        FedProx in the server), the fit method returns without doing
        training.
        This method always returns via the metrics (last argument being
        is used by strategies other than FedProx to discard the update.
        returned) whether the client is a straggler or not. This info
        """
        self.set_parameters(parameters)

        if (
            self.straggler_schedule[int(config["curr_round"]) - 1]
            and self.num_epochs > 1
        ):
            num_epochs = np.random.randint(1, self.num_epochs)

            if config["drop_client"]:
                # return without doing any training.
                # The flag in the metric will be used to tell the strategy
                # to discard the model upon aggregation
                return (
                    self.get_parameters({}),
                    len(self.trainloader),
                    {"is_straggler": True},
                )

        else:
            num_epochs = self.num_epochs


        if self.trainloader is None:

            trainset = load_data(
                client_names=[self.client_name],
                train_test_split=self.train_test_split,
                dataset_name=self.dataset_name.lower(),
                type="train",
                min_no_samples=self.min_num_samples,
                is_embedded=self.is_embedded)
            self.trainloader = DataLoader(
                trainset, batch_size=self.batch_size, shuffle=True)

        train(
            self.net,
            self.trainloader,
            self.device,
            epochs=num_epochs,
            learning_rate=self.learning_rate,
            proximal_mu=config["proximal_mu"],
        )

        return self.get_parameters({}), len(self.trainloader), {"is_straggler": False}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implements distributed evaluation for a given client."""
        self.set_parameters(parameters)
        if self.testloader is None:
            testset = load_data(
                client_names=[self.client_name],
                train_test_split=self.train_test_split,
                dataset_name=self.dataset_name.lower(),
                type="test",
                min_no_samples=self.min_num_samples,
                is_embedded=self.is_embedded)
            self.testloader = DataLoader( testset,self.batch_size,shuffle=False)

        loss, accuracy = test(self.net, self.testloader, self.device)
        return float(loss), len(self.testloader), {"accuracy": float(accuracy)}

def get_fed_client_fn(
    client_names: List[str],
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    dataset_name: str,
    learning_rate: float,
    stragglers: float,
    model: DictConfig,
    train_test_split: float,
    batch_size: int = 32,
    min_num_samples: int = 10,
):
    # Defines a staggling schedule for each clients, i.e at which round will they
    # be a straggler. This is done so at each round the proportion of staggling
    # clients is respected
    stragglers_mat = np.transpose(
        np.random.choice(
            [0, 1], size=(num_rounds, num_clients), p=[1 - stragglers, stragglers]
        )
    )
    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""
        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = instantiate(model).to(device)
        client_name = client_names[int(cid)]

        return FedFlowerClient(
            net,
            client_name,
            dataset_name,
            train_test_split,
            device,
            num_epochs,
            learning_rate,
            stragglers_mat[int(cid)],
            batch_size,
            min_num_samples
        )
    return client_fn
