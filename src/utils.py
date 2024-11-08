"""Contains utility functions."""

import os
import time
import pickle
from pathlib import Path
from secrets import token_hex
from typing import Dict, Optional, Union, Callable
import importlib.util

import matplotlib.pyplot as plt
import numpy as np
from flwr.server.history import History
from src import PATH_src, PORT_ROOT
'''
def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # Set seed for CUDA if available
'''

def server_address(ip:str,host:int)->str:
    return ''.join([ip,":",str(host)])

def timeit(func: Callable)-> Callable:
    def wrapper(self, *args, **kwargs):
        start = time.time()
        parameters,size, metrics = func(self, *args, **kwargs)
        self.time = time.time() - start
        metrics["time"] = self.time
        return parameters, size, metrics

    return wrapper


def plot_metric_from_history(
    hist: History,
    save_plot_path: Path,
    suffix: Optional[str] = "",
) -> None:
    """Function to plot from Flower server History.

    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    save_plot_path : Path
        Folder to save the plot to.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    """
    metric_type = "centralized"
    metric_dict = (
        hist.metrics_centralized
        if metric_type == "centralized"
        else hist.metrics_distributed
    )
    rounds, values = zip(*metric_dict["accuracy"])

    # let's extract centralised loss (main metric reported in FedProx paper)
    rounds_loss, values_loss = zip(*hist.losses_centralized)

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex="row")
    axs[0].plot(np.asarray(rounds_loss), np.asarray(values_loss))
    axs[1].plot(np.asarray(rounds_loss), np.asarray(values))

    axs[0].set_ylabel("Loss")
    axs[1].set_ylabel("Accuracy")

    # plt.title(f"{metric_type.capitalize()} Validation - MNIST")
    plt.xlabel("Rounds")
    # plt.legend(loc="lower right")

    plt.savefig(Path(save_plot_path) / Path(f"{metric_type}_metrics{suffix}.png"))
    plt.close()

def save_results_as_pickle(
    history: History,
    file_path: Union[str, Path],
    extra_results: Optional[Dict] = {},
    default_filename: Optional[str] = "results.pkl",
) -> None:
    """Saves results from simulation to pickle.

    Parameters
    ----------
    history: History
        History returned by start_simulation.
    file_path: Union[str, Path]
        Path to file to create and store both history and extra_results.
        If path is a directory, the default_filename will be used.
        path doesn't exist, it will be created. If file exists, a
        randomly generated suffix will be added to the file name. This
        is done to avoid overwritting results.
    extra_results : Optional[Dict]
        A dictionary containing additional results you would like
        to be saved to disk. Default: {} (an empty dictionary)
    default_filename: Optional[str]
        File used by default if file_path points to a directory instead
        to a file. Default: "results.pkl"
    """

    path = Path(file_path)

    # ensure path exists
    path.mkdir(exist_ok=True, parents=True)

    def _add_random_suffix(path_: Path):
        """Adds a randomly generated suffix to the file name (so it doesn't
        overwrite the file)."""
        print(f"File `{path_}` exists! ")
        suffix = token_hex(4)
        print(f"New results to be saved with suffix: {suffix}")
        return path_.parent / (path_.stem + "_" + suffix + ".pkl")

    def _complete_path_with_default_name(path_: Path):
        """Appends the default file name to the path."""
        print("Using default filename")
        return path_ / default_filename

    if path.is_dir():
        path = _complete_path_with_default_name(path)

    if path.is_file():
        # file exists already
        path = _add_random_suffix(path)

    print(f"Results will be saved into: {path}")

    data = {"history": history, **extra_results}

    # save results to pickle
    with open(str(path), "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def importer(filename:str):
    module_path = os.path.join(PATH_src['Models'], filename+'.py') # Specify path to the file you want to import from
    spec = importlib.util.spec_from_file_location(filename, module_path)  # Load the module from the specified path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_ports(num_clients:int,
              )->Dict[str,int]:
    return {str(cid): ( PORT_ROOT + 5 * cid) for cid in range(num_clients)}

"""
def get_communicators(num_clients:int,
                      ip_address:str,
                      ports:Dict[str, int],
                      )->Dict[str,Communicator]:
    #instantiate communicators
    communicators={}
    for cid, port in ports.items():
        communicators[cid] = Communicator(
            ip_address=ip_address,
            index=ports[cid]
        )
    return communicators

"""

