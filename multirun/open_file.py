from typing import List,Tuple
import pickle
import os
from flwr.server.history import History
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))
os.chdir(PROJECT_PATH)

from src import PATH

save_plot_path = PATH["outputs_paper"]
metric_type="loss+acc"
suffix=""

def plot_metric_from_history(
    hist: History,
    axs: plt.Axes,
    offload: bool,
    title:str=""
) -> None:
    """Function to plot from Flower server History.

    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    axs:
        Object containing subplots as in Matplotlib
    """
    color = "g" if offload else "r"
    label = "SFD" if offload else "Basic"
    metric_type = "centralized"
    metric_dict = (
        hist.metrics_centralized
        if metric_type == "centralized"
        else hist.metrics_distributed
    )
    rounds, values = zip(*metric_dict["accuracy"])
    rounds_loss, values_loss = zip(*hist.losses_centralized)
    axs[0].plot(np.asarray(rounds_loss), np.asarray(values_loss),color=color, label=label)
    axs[1].plot(np.asarray(rounds_loss), np.asarray(values),color=color, label=label)
    if len(title)>0:
        axs[0].set_title(title)


PATHS_TO_EXAMINE: List[Tuple[str,str,str]]=[
    ("mu2fstrag0.0","fedoffload","multirun/2023-09-19/11-20-19/0/results.pkl"),
    ("mu2fstrag0.5","fedoffload","multirun/2023-09-18/21-16-36/1/results.pkl"),
    ("mu2fstrag0.7","fedoffload","multirun/2023-09-18/21-16-36/2/results.pkl"),
    ("mu2fstrag0.9","fedoffload","multirun/2023-09-19/04-26-23/0/results.pkl"),
    ("mu2fstrag0.0","fedprox","multirun/2023-09-18/22-34-48/0/results.pkl"),
    ("mu2fstrag0.5","fedprox","multirun/2023-09-18/22-34-48/1/results.pkl"),
    ("mu2fstrag0.7","fedprox","multirun/2023-09-18/22-34-48/2/results.pkl"),
    ("mu2fstrag0.9","fedprox","multirun/2023-09-18/22-34-48/3/results.pkl"),
]

#Load the pickle files
experiments={}
for experiment, stroffload, path in PATHS_TO_EXAMINE:
    suffix = experiment[2]
    pkl_file_path = os.path.join(PROJECT_PATH,path)
    try:
        # Open the .pkl file in binary read mode
        with open(pkl_file_path, 'rb') as file:
            # Load the data from the .pkl file
            experiments[experiment[-3:] + stroffload ] = pickle.load(file)

        # Now, you can work with the loaded data
        print(f"Data loaded successfully: {stroffload} for settings {experiment}")
        #print(loaded_data)

    except FileNotFoundError:
        print(f"The file '{pkl_file_path}' was not found.")
    except pickle.PickleError as pe:
        print(f"Error loading data from '{pkl_file_path}': {pe}")
    except Exception as e:
        print(f"An error occurred: {e}")

fig, axs = plt.subplots(nrows=2, ncols=4, sharex="row",figsize=(30,6))

for experiment, hist in experiments.items():
    if experiment[:3]=="0.0":
        col=0
        title = "0% stragglers"
    elif experiment[:3]=="0.5":
        col=1
        title = "50% stragglers"
    elif experiment[:3]=="0.7":
        col=2
        title = "70% stragglers"
    elif experiment[:3]=="0.9":
        col=3
        title="90% stragglers"
    else: raise NotImplementedError("Needs changing the plot for meaningful results")
    print(experiment)
    offload:bool = (experiment[3:]== "fedoffload")
    plot_metric_from_history(hist['history'],axs[:,col], offload=offload,title=title)

axs[0,0].set_ylabel("Loss")
axs[1,0].set_ylabel("Accuracy")
for ax in axs[1]:
    ax.set_xlabel("Rounds")

fig.suptitle(f"FedProx with and without SFD for MNIST dataset", fontsize=20, y=1.0)
#plt.xlabel("Rounds")
plt.legend(loc="lower right",fontsize="large")
#plt.show()
plt.savefig(Path(save_plot_path) / Path(f"{metric_type}_mu_{suffix}.png"))
plt.close()
