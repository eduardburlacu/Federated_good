import os
import sys
from logging import INFO
import flwr as fl
from flwr.common.logger import log
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

#-------Insert main project directory so that we can resolve the src imports-------
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, src_path)

from src import PATH_src, DEFAULT_SERVER_ADDRESS, TIMEOUT
from src import client, server
from src.Dataset import dataset
from src.utils import get_ports
from src.utils import save_results_as_pickle,plot_metric_from_history

@hydra.main(config_path=PATH_src["conf"], config_name="config_offload", version_base=None)
def main(cfg: DictConfig) -> float:
    print(OmegaConf.to_yaml(cfg))

    # Instantiate strategy requirements according to config
    ports = get_ports(cfg.num_clients)

    #init_stragglers = {str(cid): round(cid / cfg.num_clients) for cid in range(cfg.num_clients)}
    init_capacities = {str(cid): 0.0 for cid in range(cfg.num_clients)}

    # Set server's device
    device = cfg.server_device

    #Data loading
    if cfg.dataset.lower() == "mnist":
        """
        POWER LAW PARTITIONING WITH MNIST DATASET
        """

        trainloaders, valloaders, testloader= dataset.load_datasets(
            config=cfg.dataset_config,
            num_clients=cfg.num_clients,
            batch_size=cfg.batch_size,
        )

    elif cfg.dataset.lower() == "cifar10":
        """
        Latent Dirichlet Allocation with CIFAR10 dataset
        """
        trainloaders, valloaders, testloader= dataset.load_datasets_lda(
            num_clients=cfg.num_clients,
            batch_size=cfg.batch_size,
            alfa=cfg.alpha
        )

    elif cfg.dataset.lower() in {"shakespeare", "sent140", "nist", "synth_0_0", "synth_0.5_0.5", "synth_1_1"}:
        """
        Intrinsically federated datasets
        """
        trainloaders, valloaders, testloader = dataset.load_dataset_federated(
            config=cfg.federated_dataset_config,
            dataset_name=cfg.dataset.lower(),
            batch_size=cfg.batch_size,
        )

    else:
        raise NotImplementedError("Federated dataset not available...")

    # get function that will be executed by the strategy's evaluate() method
    evaluate_fn = server.gen_evaluate_fn(testloader, device=device, model=cfg.model)
    # get function that will be executed by the strategy's configure_fit() method
    fit_config_fn = server.get_on_fit_config(cfg.fit_config)

    # instantiate strategy
    # prepare function that will be used to spawn each client
    if cfg.offload:
        # prepare function that will be used to spawn each client
        client_fn, init_stragglers = client.gen_client_fn(
            num_clients=cfg.num_clients,
            num_rounds=cfg.num_rounds,
            num_epochs=cfg.num_epochs,
            trainloaders=trainloaders,
            valloaders=valloaders,
            learning_rate=cfg.learning_rate,
            stragglers_frac=cfg.stragglers_fraction,
            capacities=init_capacities,
            model=cfg.model,
            ip_address=DEFAULT_SERVER_ADDRESS,
            ports=ports,
        )

        strategy = instantiate(
            cfg.strategy,
            min_fit_clients=cfg.clients_per_round,
            min_available_clients=2*cfg.clients_per_round,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=fit_config_fn,
            init_stragglers=init_stragglers,
            init_capacities=init_capacities,
            ports=ports,
        )
    else:
        client_fn, init_stragglers = client.gen_client_fn(
            num_clients=cfg.num_clients,
            num_rounds=cfg.num_rounds,
            num_epochs=cfg.num_epochs,
            trainloaders=trainloaders,
            valloaders=valloaders,
            learning_rate=cfg.learning_rate,
            stragglers_frac=cfg.stragglers_fraction,
            capacities=init_capacities,
            model=cfg.model,
        )


        strategy = instantiate(
            cfg.strategy,
            min_fit_clients=cfg.clients_per_round,
            min_available_clients=2 * cfg.clients_per_round,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=fit_config_fn,
        )

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds, round_timeout=TIMEOUT),
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
        strategy=strategy,
        client_manager=instantiate(cfg.client_manager)
    )

    TOTAL_TRAIN_TIME = 0.
    AVG_STRAGGLERS_DROP = 0.
    for step in range(cfg.num_rounds):
        TOTAL_TRAIN_TIME += strategy.extra_resuts["train_time"][step][1]
        AVG_STRAGGLERS_DROP += strategy.extra_resuts["frac_failures"][step][1]
    AVG_STRAGGLERS_DROP /= cfg.num_rounds
    # Experiment completed. Now we save the results and
    # generate plots using the `history`
    print("................")
    log(INFO,history)
    log(INFO,f"Decentralised metrics: {strategy.extra_resuts}")
    log(INFO,f"TOTAL_TRAIN_TIME is {TOTAL_TRAIN_TIME}")
    log(INFO,f"AVG_STRAGGLERS_DROP is {AVG_STRAGGLERS_DROP} ")
    # Hydra automatically creates an output directory
    # Let's retrieve it and save some results there
    save_path = HydraConfig.get().runtime.output_dir

    # save results as a Python pickle using a file_path
    # the directory created by Hydra for each run
    save_results_as_pickle(history, file_path=save_path, extra_results=strategy.extra_resuts)

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
    return TOTAL_TRAIN_TIME

if __name__=='__main__':
    main()