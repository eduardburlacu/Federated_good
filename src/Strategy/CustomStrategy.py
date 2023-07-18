from collections import OrderedDict
import numpy as np
import flwr as fl
import flwr.server.strategy
from typing import Dict, List, Optional, Tuple, Callable, Union
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg


class FedCustom(flwr.server.strategy.Strategy):
    def __init__(
            self, model_class, fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2, ) -> None:
        super(FedCustom, self).__init__()
        self.model = model_class
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

    def __repr__(self): return "FedCustom"

    def initialize_parameters( self, client_manager: ClientManager ) -> Optional[Parameters]:
        net = self.model()
        ndarrays = get_parameters(net)
        return fl.common.ndarrays_to_parameters(ndarrays)