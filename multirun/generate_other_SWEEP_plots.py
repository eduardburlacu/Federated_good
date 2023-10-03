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
metric_type="stragglers"
suffix="0"

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
    ("mu0fstrag0.0", "fedoffload", "multirun/fedoffload_cifar/0/results.pkl"),
    ("mu0fstrag0.1", "fedoffload", "multirun/fedoffload_cifar/1/results.pkl"),
    ("mu0fstrag0.2", "fedoffload", "multirun/fedoffload_cifar/2/results.pkl"),
    ("mu0fstrag0.3", "fedoffload", "multirun/fedoffload_cifar/3/results.pkl"),
    ("mu0fstrag0.4", "fedoffload", "multirun/fedoffload_cifar/4/results.pkl"),
    ("mu0fstrag0.5", "fedoffload", "multirun/fedoffload_cifar/5/results.pkl"),
    ("mu0fstrag0.6", "fedoffload", "multirun/fedoffload_cifar/6/results.pkl"),
    ("mu0fstrag0.7", "fedoffload", "multirun/fedoffload_cifar/7/results.pkl"),
    ("mu0fstrag0.8", "fedoffload", "multirun/fedoffload_cifar/8/results.pkl"),
    ("mu0fstrag0.9", "fedoffload", "multirun/fedoffload_cifar/9/results.pkl"),

    ("mu0fstrag0.0", "fedprox",    "multirun/fedprox_cifar/0/results.pkl"),
    ("mu0fstrag0.1", "fedprox",    "multirun/fedprox_cifar/1/results.pkl"),
    ("mu0fstrag0.2", "fedprox",    "multirun/fedprox_cifar/2/results.pkl"),
    ("mu0fstrag0.3", "fedprox",    "multirun/fedprox_cifar/3/results.pkl"),
    ("mu0fstrag0.4", "fedprox",    "multirun/fedprox_cifar/4/results.pkl"),
    ("mu0fstrag0.5", "fedprox",    "multirun/fedprox_cifar/5/results.pkl"),
    ("mu0fstrag0.6", "fedprox",    "multirun/fedprox_cifar/6/results.pkl"),
    ("mu0fstrag0.7", "fedprox",    "multirun/fedprox_cifar/7/results.pkl"),
    ("mu0fstrag0.8", "fedprox",    "multirun/fedprox_cifar/8/results.pkl"),
    ("mu0fstrag0.9", "fedprox",    "multirun/fedprox_cifar/9/results.pkl"),
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
fig, axs = plt.subplots(nrows=2, ncols=1, sharex="row",figsize=(12,12))

total_train_time_without = []
total_train_time_with = []
avg_straggler_drop_without= []
avg_straggler_drop_with= []
stragglers =[i/10 for i in range(10)]

for experiment, (histoffload, histbasic) in experiments.items():
    if experiment[:3]=="0.0":
        strag =0.0
    elif experiment[:3]=="0.1":
        strag = 0.1
    elif experiment[:3]=="0.2":
        strag = 0.2
    elif experiment[:3]=="0.3":
        strag = 0.3
    elif experiment[:3]=="0.4":
        strag = 0.4
    elif experiment[:3] == "0.5":
        strag = 0.5
    elif experiment[:3] == "0.6":
        strag = 0.6
    elif experiment[:3] == "0.7":
        strag = 0.7
    elif experiment[:3] == "0.8":
        strag = 0.8
    elif experiment[:3] == "0.9":
        strag = 0.9
    else: raise NotImplementedError("Needs changing the plot for meaningful results")

    #print(f"{experiment} got as train_time results,{hist['train_time']}")
    #print(f"{experiment} got as frac_failures results,{hist['frac_failures']}")
    #print(f"{experiment} got as TOTAL_TRAIN_TIME results,{hist['TOTAL_TRAIN_TIME']}")
    #print(f"{experiment} got as AVG_STRAGGLERS_DROP results,{hist['AVG_STRAGGLERS_DROP']}")
    total_train_time_with.append(histoffload['TOTAL_TRAIN_TIME'])
    total_train_time_without.append(histbasic['TOTAL_TRAIN_TIME'])
    avg_straggler_drop_with.append(histoffload['AVG_STRAGGLERS_DROP'])
    avg_straggler_drop_without.append(histbasic['AVG_STRAGGLERS_DROP'])

#plot_metric_from_history(,axs, title=title)
axs[0].plot(stragglers, total_train_time_with, c='g',label="SFD")
axs[0].plot(stragglers, total_train_time_without, c='r', label="Basic")

axs[0].set_ylabel("Training time(s)")
axs[0].grid()
#plot_metric_from_history(,axs, title=title)
axs[1].plot(stragglers, avg_straggler_drop_with, c='g',label="SFD")
axs[1].plot(stragglers, avg_straggler_drop_without, c='r', label="Basic")

axs[1].set_xlabel("Fraction of strragglers(%)")
axs[1].set_ylabel("Average Straggler Drop(%)")
axs[1].grid()

fig.suptitle(f"FedAvg with and without SFD for CIFAR10 dataset", fontsize=20, y=1.0)
fig.tight_layout()

axs[0].legend(loc="best",fontsize="large")
#plt.show()
plt.savefig(Path(save_plot_path) / Path(f"{metric_type}_mu_{suffix}.png"))
plt.close()
