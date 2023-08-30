import os
import sys
import flwr as fl
from flwr.server.client_manager import SimpleClientManager
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

#-------Insert main project directory so that we can resolve the src imports-------
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, src_path)

from src import PATH_src, DEFAULT_SERVER_ADDRESS
from src import client, server
from src.ClientManager import OffloadClientManager
from src.Dataset import dataset
from src.utils import save_results_as_pickle,plot_metric_from_history

@hydra.main(config_path=PATH_src["conf"], config_name="config_offload", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    if cfg.dataset.lower() in {"mnist", "cifar10"}:
        trainloaders, valloaders, testloader, datasizes = dataset.load_datasets(
            config=cfg.dataset_config,
            num_clients=cfg.num_clients,
            batch_size=cfg.batch_size,
        )
        # prepare function that will be used to spawn each client
        client_fn = client.gen_client_fn(
            num_clients=cfg.num_clients,
            num_rounds=cfg.num_rounds,
            num_epochs=cfg.num_epochs,
            trainloaders=trainloaders,
            valloaders=valloaders,
            learning_rate=cfg.learning_rate,
            stragglers=cfg.stragglers_fraction,
            model=cfg.model,
            ip_address=DEFAULT_SERVER_ADDRESS,
        )

        # get function that will be executed by the strategy's evaluate() method
        # Set server's device
        device = cfg.server_device
        evaluate_fn = server.gen_evaluate_fn(testloader, device=device, model=cfg.model)

        #Instantiate agent accoridng to config
        agent = instantiate(cfg.agent)
        # instantiate strategy according to config
        strategy = instantiate(
            cfg.strategy,
            evaluate_fn=evaluate_fn,
            agent=agent
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
            client_manager=OffloadClientManager()
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

        plot_metric_from_history(
            history,
            save_path,
            (file_suffix),
        )

if __name__=='__main__':
    main()
