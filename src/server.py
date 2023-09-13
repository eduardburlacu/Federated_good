from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple, List

import torch
from flwr.common import Metrics
from flwr.common.typing import NDArrays, Scalar, Metrics
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.models import test

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


def fit_metrics_aggregation_fn( metrics:List[Tuple[int,Metrics]] ) -> Metrics:
    longest_train_time = 0.0

    for num_examples, m in metrics:
        longest_train_time = max(
            longest_train_time,
            m["time"]
        )

    return {"train_time": longest_train_time}