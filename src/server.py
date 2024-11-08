from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple, List

import torch
from torch.utils.data import DataLoader

from flwr.common.typing import NDArrays, Scalar, Metrics
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from src.models import test
from src import SEED

torch.manual_seed(SEED)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(SEED)  # Set seed for CUDA if available
torch.use_deterministic_algorithms(True)

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for weighted average during evaluation.

    Parameters
    ----------
    metrics : List[Tuple[int, Metrics]]
        The list of metrics to aggregate.

    Returns
    -------
    Metrics
        The weighted average metric.
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": int(sum(accuracies)) / int(sum(examples))}

def fit_metrics_aggregation_fn( metrics:List[Tuple[int,Metrics]] ) -> Metrics:
    longest_train_time = 0.0

    for num_examples, m in metrics:
        longest_train_time = max(
            longest_train_time,
            m["time"]
        )

    return {"train_time": longest_train_time}


def get_on_fit_config(conf):
    def fit_config_fn(server_round: int):
        # resolve and convert to python dict
        fit_config: Dict[str, Union[bool, float]] = OmegaConf.to_container(  # type: ignore
            conf, resolve=True
        )
        fit_config["curr_round"] = server_round  # add round info
        return fit_config
    return fit_config_fn

def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    model: DictConfig,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Generates the function for centralized evaluation.

    Parameters
    ----------
    testloader : DataLoader
        The dataloader to test the model with.
    device : torch.device
        The device to test the model on.
    model : DictConfig
        The model class to be used.

    Returns
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]] ]
        The centralized evaluation function.
    """
    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # pylint: disable=unused-argument
        """Use the entire CIFAR-10 test set for evaluation."""

        net = instantiate(model)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        loss, accuracy = test(net, testloader, device=device)
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate
