import time
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple, Optional
from math import ceil
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src import SEED
from src.Dataset.dataset_preparation_federated import load_data
from src.models import test, train
from src.Communication import Communicator
from src.straggler_schedule import get_straggler_schedule
from src.split_learn import split_model
from src.utils import timeit

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED) # Set seed for CUDA if available
torch.use_deterministic_algorithms(True)

class FlowerClient(
    fl.client.NumPyClient,
    Communicator
):  # pylint: disable=too-many-instance-attributes
    beta:float = 0.85
    def __init__(
        self,
        cid:str,
        model:DictConfig,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        straggler_schedule: np.ndarray,
        computation_frac: Optional[float],  # What fraction of the train load a device can solve
        ip_address:Optional[str],           # Communication address
        index:Optional[int],                # Communication host

    ):  # pylint: disable=too-many-arguments
        '''
        Implements a unique FL participant.

        :param cid: Unique identifier as str(integer)
        :param model: Neural Network class to be instantiated for neural network
        :param trainloader: Torch Dataloader for training
        :param valloader: Torch Dataloader for validation
        :param device: Torch device for training
        :param num_epochs: Number of local epochs to be ideally performed locally(on client)
        :param learning_rate: Local learning rate
        :param straggler_schedule: Sets a random schedule in which a device is a straggler with probability p, performing a random fraction of training
        :param ip_address: Client socket ip address (defaulted as the local address for a local simulation)
        :param index: Port number for client socket
        '''
        self.cid = cid
        self.model =model
        self.net = instantiate(self.model).to(device)

        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device

        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.straggler_schedule = straggler_schedule
        self.computation_frac = computation_frac

        self.capacity = 0. #updated at the end of initilaizer

        self.time= 0.

        if index is None or ip_address is None:
            self.connection_failure = True
            self.mbps=0.
        else:
            # In case the communication is unavailable,
            # we let the straggler train alone.
            try:
                Communicator.__init__(
                    self,
                    ip_address=ip_address,
                    index=index,
                )
                self.connection_failure = False

            except:
                self.connection_failure = True
                self.mbps=0.

        self.update_capacity()

    def get_properties(self, config: Dict[str, Scalar]) -> Dict[str, Scalar]:
        #Update capacity
        properties = {
            "time": self.time,
            "capacity":self.capacity
        }
        return properties

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Returns the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Changes the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def update_capacity(self):
        target = self.mbps * ceil(len(self.trainloader)/self.trainloader.batch_size) / self.computation_frac
        self.capacity = (1 - FlowerClient.beta) * self.capacity + FlowerClient.beta * target
    def split_train(
            self,
            split_layer:int,
            num_epochs:int,
            frac: float = 1.0,
            momentum: float = 0.9
    )->None:

        if self.net:
            self.net = split_model(net=self.net, n1= split_layer)[0]

        else: raise RuntimeError('Model missing in split train.')

        global_parms = [val.detach().clone() for val in self.net.parameters()]
        self.net.train()

        if split_layer > 0:
            optimizer = torch.optim.SGD(
                self.net.parameters(),
                lr=self.learning_rate,
                momentum=momentum,
            )
            for e in range(num_epochs):
                for batch_idx, (features, targets) in enumerate(self.trainloader):
                    dt = time.time()
                    features, targets = features.to(self.device), targets.to(self.device)
                    proximal_term = 0.0
                    optimizer.zero_grad()
                    for local_weights, global_weights in zip(self.net.parameters(), global_parms):
                        proximal_term += (local_weights - global_weights).norm(2)
                    smashed_activations = self.net(features)
                    # Transfer data to other device
                    msg = ["MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER",
                           split_layer,
                           proximal_term,
                           smashed_activations.cpu(),
                           targets.cpu(),
                           ""
                           ]

                    dt = (time.time()-dt) * (1/frac - 1)
                    #Communicator is separated
                    self.send_msg(
                        sock=self.to_socket(),
                        msg=msg,
                    )
                    time.sleep(dt)
                    self.update_capacity()
                    # Wait for backpropagation if split learning
                    gradients = self.recv_msg(self.to_socket())[1]
                    if split_layer > 0:
                        gradients = gradients.to(self.device)
                        smashed_activations.backward(gradients)
                    msg = ['MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER',
                           self.cid,
                           self.get_parameters({}),
                           len(self.trainloader),
                           self.capacity
                           ]
                    self.send_msg(sock=self.to_socket(), msg=msg)
                    self.update_capacity()

        else: #Direct offload
            # Transfer data to other device
            msg = ["MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER",
                   self.trainloader,
                   self.cid,
                   self.capacity
                   ]
            self.send_msg(
                sock=self.to_socket(),
                msg=msg,
            )
            self.update_capacity()

    def split_follower(self,
                       proximal_mu:float,
                       momentum: float = 0.9,
                       frac: float = 1.0,
    ):
        # Wait for forward prop
        msg = self.recv_msg(
            sock= self.to_socket(),
            expect_msg_type='MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER'
        )

        if len(msg)==5:
            cid, ante_parameters ,num_examples, capacity = msg[1:]
            return ([*ante_parameters, *self.get_parameters({})],
                    num_examples,
                    { "is_straggler": self.computation_frac != 1.0,
                      "cid": cid,
                      "capacity":capacity,
                      }
                    )

        elif len(msg)==4:
            trainloader, cid, capacity = msg[1:]
            dt = time.time()
            train(
                self.net,
                trainloader= trainloader,
                device=self.device,
                epochs=self.num_epochs,
                learning_rate=self.learning_rate,
                proximal_mu=proximal_mu,
                computation_frac=frac,
            )
            dt = time.time() - dt
            time.sleep(dt)
            return self.get_parameters({}), len(trainloader), {
                "is_straggler": self.computation_frac != 1.0,
                "cid": cid,
                "capacity":capacity
            }

        else:
            criterion = nn.CrossEntropyLoss()
            split_layer, proximal_term_straggler, smashed_activations, targets, _ = msg[1:]
            dt = time.time()
            # Split training
            if self.net:
                self.net = split_model(net=self.net, n1=split_layer)[1]

            else: raise RuntimeError('Model missing in split train.')

            self.net.train()
            global_params = [val.detach().clone() for val in self.net.parameters()]

            optimizer = torch.optim.SGD(
                self.net.parameters(),
                lr=self.learning_rate,
                momentum=momentum,
            )
            smashed_activations, targets = smashed_activations.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            proximal_term_follower = 0.0
            for local_weights, global_weights in zip(self.net.parameters(), global_params):
                proximal_term_follower += (local_weights - global_weights).norm(2)
            output = self.net(smashed_activations)
            proximal_term = proximal_term_straggler + proximal_term_follower
            loss = criterion(output, targets) + proximal_mu * proximal_term /2
            loss.backward()
            optimizer.step()
            dt = (time.time()-dt) * (1/frac - 1)
            # Send gradients to client
            msg = [
                'MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT',
                smashed_activations.grad
            ]
            self.send_msg(sock=self.to_socket(), msg=msg)
            time.sleep(dt)
            self.update_capacity()
            return None

    @timeit
    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implements distributed fit function for a given client."""
        self.update_capacity()
        self.set_parameters(parameters)
        if (
            self.straggler_schedule[int(config["curr_round"]) - 1]
            and self.num_epochs > 1
        ):
            if "drop_client" in config:
                if config["drop_client"]:
                    # return without doing any training.
                    return (
                        self.get_parameters({}),
                        len(self.trainloader),
                        {"is_straggler": True,
                         "cid":self.cid,
                         "next": self.straggler_schedule[min(config["curr_round"], len(self.straggler_schedule)-1)],
                         "capacity": self.capacity
                         }
                    )

        else:
            self.computation_frac = 1.0

        if "follower" in config:
            # Wait for connection to straggler(s)
            if not self.connected:
                self.listen(config["follower"])
            #Update network from server
            if len(list(self.net.children())) != len(parameters):
                self.net = instantiate(self.model).to(self.device)
            self.set_parameters(parameters)

            result = None
            while result is None:
                result = self.split_follower(proximal_mu=config["proximal_mu"], frac=self.computation_frac)
            print(f'Follower {self.cid}, flops {self.computation_frac} has finished training round')

            return result

        else:
            query = self.connection_failure or (config["split_layer"] == len(list(self.net.children())) - 1)
            if query:  # Independent training
                train(
                    self.net,
                    self.trainloader,
                    self.device,
                    epochs=self.num_epochs,
                    learning_rate=self.learning_rate,
                    proximal_mu=config["proximal_mu"],
                    computation_frac=self.computation_frac
                )
                print(f'Regular client {self.cid}, flops {self.computation_frac} has finished training')
                return self.get_parameters({}), len(self.trainloader), {
                    "is_straggler": self.computation_frac!=1.0,
                    "cid": self.cid,
                    "next": self.straggler_schedule[min(
                        config["curr_round"],
                        len(self.straggler_schedule)-1
                    )],
                    "capacity":self.capacity
                }

            else:  # Offload a part of the training
                # Wait for connection
                
                self.connect(
                    other_addr=self.ip,
                    other_port=config["port"]
                )
                if len(list(self.net.children())) != len(parameters):
                    self.net = instantiate(self.model).to(self.device)

                self.set_parameters(parameters)

                self.split_train(
                    split_layer=config["split_layer"],
                    num_epochs=self.num_epochs,
                    frac=self.computation_frac
                )

                print(f'Straggler {self.cid}, flops {self.computation_frac} has finished training round')
                return ([], len(self.trainloader), {
                    "is_straggler": True,
                    "cid": self.cid,
                    "next": self.straggler_schedule[min(config["curr_round"], len(self.straggler_schedule)-1)],
                    "capacity": self.capacity
                })


    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implements distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def gen_client_fn(
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    learning_rate: float,
    stragglers_frac: float,
    model: DictConfig,
    ip_address:Optional[str]=None,
    ports:Optional[Dict[str,int]]=None,
) -> Tuple[
     Callable[[str], FlowerClient],
    Dict[str, bool],
    Dict[str, bool]
]:  # pylint: disable=too-many-arguments
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
    stragglers_frac : float
        Proportion of stragglers in the clients, between 0 and 1.

    Returns
    -------
    Tuple[Callable[[str], FlowerClient], DataLoader]
        A tuple containing the client function that creates Flower Clients and
        the DataLoader that will be used for testing
    """

    # Defines a staggling schedule for each client, i.e at who and at which round will
    # be a straggler and how much it can perform in a given time.
    # This is done so at each round the proportion of staggling clients is respected

    stragglers_mat, computation_fracs = get_straggler_schedule(
        num_clients=num_clients,
        num_rounds=num_rounds,
        stragglers_frac=stragglers_frac,
        type="constant",
    )
    init_stragglers = {str(cid): bool(stragglers_mat[cid,0]) for cid in range(num_clients)}

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        # Load data and device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]
        # Allow offload=false option
        if ports is None:
            index = None
        else:
            index = ports[cid]

        return FlowerClient(
            cid=cid,
            model=model,
            trainloader=trainloader,
            valloader=valloader,
            device=device,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            straggler_schedule=stragglers_mat[int(cid)],
            computation_frac=computation_fracs[cid],
            ip_address=ip_address,
            index= index
        )

    return client_fn, init_stragglers,computation_fracs

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
        self.is_embedded = self.dataset_name == "shakespeare"

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implements distributed fit function for a given client."""
        self.set_parameters(parameters)


        # At each round check if the client is a straggler,
        # if so, train less epochs (to simulate partial work)
        # if the client is told to be dropped (e.g. because not using
        # FedProx in the server), the fit method returns without doing
        # training.
        # This method always returns via the metrics (last argument being
        # returned) whether the client is a straggler or not. This info
        # is used by strategies other than FedProx to discard the update.
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
