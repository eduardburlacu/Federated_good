"""Runs CNN federated learning for MNIST dataset."""
import os
import sys
import flwr as fl
import hydra
import torch.utils.data
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

#----Insert main project directory so that we can resolve the src imports-------
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, src_path)

from src import PATH_src, GOD_CLIENT_NAME
from src import client, server, utils, dataset, federated_dataset
from src.utils import save_results_as_pickle

@hydra.main(config_path=PATH_src["conf"], config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function to run CNN federated learning on MNIST.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # print config structured as YAML
    print(OmegaConf.to_yaml(cfg))
    if cfg.dataset.lower() in {"mnist","cifar10"}:
        # partition dataset and get dataloaders
        trainloaders, valloaders, testloader = dataset.load_datasets(
            config=cfg.dataset_config,
            num_clients=cfg.num_clients,
            batch_size=cfg.batch_size,
        )

        # prepare function that will be used to spawn each client
        client_fn = client.gen_client_fn(
            num_clients=cfg.num_clients,
            num_epochs=cfg.num_epochs,
            trainloaders=trainloaders,
            valloaders=valloaders,
            num_rounds=cfg.num_rounds,
            learning_rate=cfg.learning_rate,
            stragglers=cfg.stragglers_fraction,
            model=cfg.model,
        )


    elif cfg.dataset in {"sent140","shakespeare","nist","synthetic_0.5_0.5","synthetic_0_0",'synthetic_1_1',"synthetic_iid",}:
        testset = federated_dataset.load_data(
            client_names=[GOD_CLIENT_NAME],
            train_test_split=0.9,
            dataset_name=cfg.dataset,
            type="test",
            min_no_samples=cfg.federated_dataset_config.min_dataset_size,
            is_embedded=cfg.federated_dataset_config.is_embedded,
        )
        testloader = torch.utils.data.DataLoader(testset, cfg.batch_size, shuffle=False)
        print("centralized testset length: ", len(testset))

        client_fn = client.get_fed_client_fn(
            num_clients=cfg.num_clients,
            num_rounds=cfg.num_rounds,
            num_epochs=cfg.num_epochs,
            dataset_name=cfg.dataset,
            learning_rate=cfg.learning_rate,
            stragglers=cfg.stragglers_fraction,
            model=cfg.model,
            train_test_split=cfg.federated_dataset_config.train_test_split,
            batch_size=cfg.batch_size,
            min_num_samples=cfg.federated_dataset_config.min_num_samples,
        )

    else: raise ValueError('Dataset in configuration not available.')

    # get function that will executed by the strategy's evaluate() method
    # Set server's device
    device = cfg.server_device
    evaluate_fn = server.gen_evaluate_fn(testloader, device=device, model=cfg.model)

    # get a function that will be used to construct the config that the client's
    # fit() method will received
    def get_on_fit_config():
        def fit_config_fn(server_round: int):
            # resolve and convert to python dict
            fit_config = OmegaConf.to_container(cfg.fit_config, resolve=True)
            fit_config["curr_round"] = server_round  # add round info
            return fit_config

        return fit_config_fn

    # instantiate strategy according to config. Here we pass other arguments
    # that are only defined at run time.
    strategy = instantiate(
        cfg.strategy,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=get_on_fit_config(),
    )

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
        strategy=strategy,
    )

    # Experiment completed. Now we save the results and
    # generate plots using the `history`
    print("................")
    print(history)

    # Hydra automatically creates an output directory
    # Let's retrieve it and save some results there
    save_path = HydraConfig.get().runtime.output_dir

    # save results as a Python pickle using a file_path
    # the directory created by Hydra for each run
    save_results_as_pickle(history, file_path=save_path, extra_results={})

    # plot results and include them in the readme
    strategy_name = strategy.__class__.__name__
    file_suffix: str = (
        f"_{strategy_name}"
        f"{'_iid' if cfg.dataset_config.iid else ''}"
        f"{'_balanced' if cfg.dataset_config.balance else ''}"
        f"{'_powerlaw' if cfg.dataset_config.power_law else ''}"
        f"_C={cfg.num_clients}"
        f"_B={cfg.batch_size}"
        f"_E={cfg.num_epochs}"
        f"_R={cfg.num_rounds}"
        f"_mu={cfg.mu}"
        f"_strag={cfg.stragglers_fraction}"
    )

    utils.plot_metric_from_history(
        history,
        save_path,
        (file_suffix),
    )


if __name__ == "__main__":
    main()
