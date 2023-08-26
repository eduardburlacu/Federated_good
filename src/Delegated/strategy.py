
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    Metrics,
    #EvaluateIns,
    #EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.strategy.aggregate import weighted_loss_avg,aggregate
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.Delegated.ClientManager import CustomClientManager

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
    print("here and nothing is breaking!!!")
    return {"accuracy": int(sum(accuracies)) / int(sum(examples))}

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

# flake8: noqa: E501
class FedProx(FedAvg):
    """Configurable FedProx strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        proximal_mu: float= 0.,
        agent
    ) -> None:
        """Federated Optimization strategy.

        Implementation based on https://arxiv.org/abs/1812.06127

        The strategy in itself will not be different than FedAvg, the client needs to be adjusted.
        A proximal term needs to be added to the loss function during the training:

        .. math::
            \\frac{\\mu}{2} || w - w^t ||^2

        Where $w^t$ are the global parameters and $w$ are the local weights the function will
        be optimized with.

        In PyTorch, for example, the loss would go from:

        .. code:: python

          loss = criterion(net(inputs), labels)

        To:

        .. code:: python

          for local_weights, global_weights in zip(net.parameters(), global_params):
              proximal_term += (local_weights - global_weights).norm(2)
          loss = criterion(net(inputs), labels) + (config["proximal_mu"] / 2) * proximal_term

        With `global_params` being a copy of the parameters before the training takes place.

        .. code:: python

          global_params = copy.deepcopy(net).parameters()

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. In case `min_fit_clients`
            is larger than `fraction_fit * available_clients`, `min_fit_clients`
            will still be sampled. Defaults to 1.0.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. In case `min_evaluate_clients`
            is larger than `fraction_evaluate * available_clients`, `min_evaluate_clients`
            will still be sampled. Defaults to 1.0.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        proximal_mu : float
            The weight of the proximal term used in the optimization. 0.0 makes
            this strategy equivalent to FedAvg, and the higher the coefficient, the more
            regularization will be used (that is, the client parameters will need to be
            closer to the server parameters during training).
        """
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.proximal_mu = proximal_mu
        self.stragglers = set()
        self.capacities = {}
        self.agent = agent

    def __repr__(self) -> str:
        rep = f"FedProx(accept_failures={self.accept_failures})"
        return rep


    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: CustomClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.
        Sends the proximal factor mu to the clients
        """

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        client_manager.register()
        clients, jobs = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
            stragglers= self.stragglers,
            capacity=self.capacities,
        )

        result=[]
        for client in clients:
            config = {}
            if self.on_fit_config_fn is not None:
                # Custom fit config function provided
                config = self.on_fit_config_fn(server_round)

            config["curr_round"]= server_round
            config["proximal_mu"] = self.proximal_mu
            if client.cid in jobs:
                config["follower"] = jobs[client.cid]
                config["split_layer"] = self.agent.exploit() #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!RL INTEGRATION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            elif client.cid in self.stragglers:
                config["split_layer"] = self.agent.exploit()

            result.append(
                (client, FitIns(parameters, config))
            )

        return result

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = []

        for client_prox,fit_res in results:

            weight = parameters_to_ndarrays(fit_res.parameters)
            if fit_res.metrics["next"]==1:
                self.stragglers.add(client_prox.cid)
            if len(weight)>0:
                weights_results.append((weight, fit_res.num_examples))

        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

