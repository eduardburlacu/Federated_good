from typing import List,Tuple
from collections import OrderedDict
import pickle
import os
from flwr.server.history import History
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))
os.chdir(PROJECT_PATH)

from src import PATH, TIMEOUT

save_plot_path = PATH["outputs_paper"]
metric_type="train_time"
suffix="2"

def plot_metric_from_history(
    hist: List[Tuple[int,float]],
    axs: plt.Axes,
    title:str=""
) -> None:

    rounds, values = zip(*hist)
    axs.hist(np.asarray(values)/TIMEOUT, bins=20,range=(0,1),density=True,color=["g","r"], label=["SFD","Basic"])
    if len(title)>0:
        axs.set_title(title)

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

#------------------------------Load the pickle files--------------------------
experiments=OrderedDict()
for experiment, stroffload, path in PATHS_TO_EXAMINE:
    suffix = experiment[2]
    pkl_file_path = os.path.join(PROJECT_PATH,path)
    try:
        # Open the .pkl file in binary read mode
        with open(pkl_file_path, 'rb') as file:
            # Load the data from the .pkl file
            if experiment[-3:] not in experiments:
                experiments[ experiment[-3:] ] = [pickle.load(file)]
            else:
                experiments[ experiment[-3:]].append(pickle.load(file))

        # Now, you can work with the loaded data
        print(f"Data loaded successfully: {stroffload} for settings {experiment}")

    except FileNotFoundError:
        print(f"The file '{pkl_file_path}' was not found.")
    except pickle.PickleError as pe:
        print(f"Error loading data from '{pkl_file_path}': {pe}")
    except Exception as e:
        print(f"An error occurred: {e}")

#----------------------------------Generate plots-------------------------------
fig, axs = plt.subplots(nrows=1, ncols=4, sharex="row",figsize=(30,6))

for experiment, (histoffload, histbasic) in experiments.items():
    if experiment=="0.0":
        col=0
        title = "0% stragglers"
    elif experiment=="0.5":
        col=1
        title = "50% stragglers"
    elif experiment=="0.7":
        col=2
        title = "70% stragglers"
    elif experiment=="0.9":
        col=3
        title="90% stragglers"
    else: raise NotImplementedError("Needs changing the plot for meaningful results")

    #print(f"{experiment} got as train_time results,{hist['train_time']}")
    #print(f"{experiment} got as frac_failures results,{hist['frac_failures']}")
    #print(f"{experiment} got as TOTAL_TRAIN_TIME results,{hist['TOTAL_TRAIN_TIME']}")
    #print(f"{experiment} got as AVG_STRAGGLERS_DROP results,{hist['AVG_STRAGGLERS_DROP']}")

    data = zip(histoffload["train_time"],histbasic["train_time"])
    plot_metric_from_history(data,axs[col], title=title)

for idx,ax in enumerate(axs):
    if idx==0:
        ax.set_ylabel("Train time /Timeout")
    ax.set_xlabel("Rounds")

fig.suptitle(f"FedProx with and without SFD for MNIST dataset", fontsize=20, y=1.0)
fig.tight_layout()
plt.legend(loc="best",fontsize="large")
#plt.show()
plt.savefig(Path(save_plot_path) / Path(f"{metric_type}_mu_{suffix}.png"))
plt.close()
