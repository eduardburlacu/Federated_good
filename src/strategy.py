from typing import List, Tuple, Dict, Union, Callable, Optional
from logging import WARNING

from flwr.common.logger import log
from flwr.common import (
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

from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedProx
from flwr.server.strategy.aggregate import aggregate
from src import TIMEOUT
from src.ClientManager import OffloadClientManager
WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


class FedAvgWithStragglerDrop(FedAvg):
    """Custom FedAvg which discards updates from stragglers."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ):
        """Here we discard all the models sent by the clients that were
        stragglers in this round."""

        # Record which client was a straggler in this round
        stragglers_mask = [res.metrics["is_straggler"] for _, res in results]

        print(f"Num stragglers in round: {sum(stragglers_mask)}")

        # keep those results that are not from stragglers
        results = [res for i, res in enumerate(results) if not (stragglers_mask[i])]

        # call the parent `aggregate_fit()` (i.e. that in standard FedAvg)
        return super().aggregate_fit(server_round, results, failures)


# flake8: noqa: E501
class FedProxOffload(FedAvg):
    """Offloading FedProx strategy implementation."""
    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
        self,
        *,
        model_num_layers: int,
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
        agent=None,
        init_stragglers: Dict[str, int]=None,
        init_capacities: Dict[str,bool]=None,
        ports: Dict[str, int],
    ) -> None:
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
        self.model_num_layers = model_num_layers
        self.proximal_mu = proximal_mu
        self.agent = agent
        self.stragglers = init_stragglers
        self.capacities = init_capacities
        self.ports = ports
        self.extra_resuts={}

    def __repr__(self) -> str:
        rep = f"FedProx(offload=True, accept_failures={self.accept_failures})"
        return rep

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: OffloadClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.
        Sends the proximal factor mu to the clients
        """
        print(f"AVAILABLE FOR TRAINING {client_manager.num_available()}")
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        clients, clients_cid, jobs, ports = client_manager.sample(
            num_clients= sample_size,
            min_num_clients= min_num_clients,
            with_followers= True,
            stragglers=self.stragglers,
            capacities=self.capacities,
            ports = self.ports
        )
        print(f"CLIENTS CONVOCATED ARE: {clients_cid} AND JOBS ARE: {jobs} AND PORTS ARE {ports}")
        result=[]
        for cid, client in zip(clients_cid, clients):
            config = {}
            if self.on_fit_config_fn is not None: # Custom fit config function provided
                config = self.on_fit_config_fn(server_round)

            config["proximal_mu"] = self.proximal_mu
            config["split_layer"] = self.model_num_layers - 1

            if cid in jobs: #follower configuration
                config["follower"] = jobs[cid]
                config["split_layer"] = self.agent.exploit()
            elif cid in ports: #straggler configuration
                config["port"] = ports[cid]
                config["split_layer"] = self.agent.exploit()
                #del res

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
        # Measure what % has been dropped off
        frac_failures = len(failures)/(len(results)+len(failures))

        for client_prox,fit_res in results:
            print(f"Time, straggler={fit_res.metrics['is_straggler']}: {fit_res.metrics['time']}")
            if "next" in fit_res.metrics:
                # Update record of stragglers at the moment
                self.stragglers[fit_res.metrics["cid"]] = bool(fit_res.metrics["next"])

            weight = parameters_to_ndarrays(fit_res.parameters)
            if len(weight)>0: #Filter stragglers aided by followers
                weights_results.append((weight, fit_res.num_examples))

        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Measure what % has been dropped off
        frac_failures = len(failures)/(len(results)+len(failures))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            metrics_aggregated["frac_failures"] = frac_failures
            print(metrics_aggregated)
            # Cache results before simulation end
            for key, value in metrics_aggregated.items():
                if key not in self.extra_resuts:
                    self.extra_resuts[key] = [(server_round,value)]
                else:
                    self.extra_resuts[key].append((server_round, value))

        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

"""
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        \"\"\"Aggregate evaluation losses using weighted average.\"\"\"
        loss_aggregated, metrics_aggregated= super().aggregate_evaluate(
            server_round,
            results,
            failures,
        )
        metrics_aggregated["training_time"]= self.time_buffer
        return loss_aggregated, metrics_aggregated
"""

class FedProxNonOffload(FedProx):
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
            proximal_mu: float,
    ) -> None:
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
            proximal_mu=proximal_mu,
        )
        self.extra_resuts={}


    def __repr__(self) -> str:
        rep = f"FedProx(offload=False, accept_failures={self.accept_failures})"
        return rep

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
            print(f"Time, straggler={fit_res.metrics['is_straggler']}: {fit_res.metrics['time']}")
            weight = parameters_to_ndarrays(fit_res.parameters)
            if len(weight)>0: #Filter stragglers aided by followers
                weights_results.append((weight, fit_res.num_examples))

        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Measure what % has been dropped off
        frac_failures = len(failures)/(len(results)+len(failures))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            metrics_aggregated["frac_failures"] = frac_failures
            if frac_failures > 0:
                metrics_aggregated["train_time"] = TIMEOUT
            print(metrics_aggregated)
            # Cache results before simulation end
            for key, value in metrics_aggregated.items():
                if key not in self.extra_resuts:
                    self.extra_resuts[key] = [(server_round,value)]
                else:
                    self.extra_resuts[key].append((server_round, value))

        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
