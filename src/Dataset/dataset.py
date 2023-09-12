"""Dataset loading for federated learning."""


from typing import Optional, Tuple, List

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

from src import GOD_CLIENT_NAME
from src.Dataset.dataset_preparation_mnist import _partition_data
from src.Dataset.dataset_preparation_cifar10 import do_fl_partitioning, get_dataloader, get_cifar_10
from src.Dataset.dataset_preparation_federated import load_data, FederatedDataset
def load_datasets(  # pylint: disable=too-many-arguments
    config: DictConfig,
    num_clients: int,
    val_ratio: float = 0.1,
    batch_size: Optional[int] = 32,
    seed: Optional[int] = 42,
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader,]:
    """Creates the MNIST dataloaders to be fed into the model.

    Parameters
    ----------
    config: DictConfig
        Parameterises the dataset partitioning process
    num_clients : int
        The number of clients that hold a part of the data
    val_ratio : float, optional
        The ratio of training data that will be used for validation (between 0 and 1),
        by default 0.1
    batch_size : int, optional
        The size of the batches to be fed into the model, by default 32
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        The DataLoader for training, the DataLoader for validation, the DataLoader for testing.
    """
    print(f"Dataset partitioning config: {config}")
    datasets, testset, total_size = _partition_data(
        num_clients,
        iid=config.iid,
        balance=config.balance,
        power_law=config.power_law,
        seed=seed,
    )
    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for dataset in datasets:
        len_val = int(len(dataset) / (1 / val_ratio))
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(
            dataset, lengths, torch.Generator().manual_seed(seed)
        )
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))

    return trainloaders, valloaders, DataLoader(testset, batch_size=batch_size)

def load_datasets_lda( # pylint: disable=too-many-arguments
        num_clients: int,
        val_ratio: float = 0.1,
        batch_size: Optional[int] = 32,
        alfa:float = 1000.,
)-> Tuple[List[DataLoader], List[DataLoader], DataLoader,]:
    train_path, testset = get_cifar_10()
    fed_dir = do_fl_partitioning(
        train_path,
        pool_size=num_clients,
        alpha=alfa,
        num_classes=10,
        val_ratio=val_ratio
    )
    trainloaders=[]
    valloaders = []
    for cid in range(num_clients):
        trainloaders.append(
            get_dataloader(fed_dir,str(cid),is_train=True,batch_size=batch_size,)
        )
        valloaders.append(
            get_dataloader(fed_dir, str(cid), is_train=False, batch_size=batch_size, )
        )

    return trainloaders, valloaders, DataLoader(testset, batch_size=batch_size)

def load_dataset_federated(
        config: DictConfig,
        dataset_name:str,
        batch_size: Optional[int] = 32,
):
    testset = load_data(
        [GOD_CLIENT_NAME],
        train_test_split= config.train_test_split,
        dataset_name = dataset_name.lower(),
        type="test",
        min_no_samples=config.min_num_samples,
        is_embedded=config.is_embedded,
    )
    testloader = DataLoader(testset,batch_size,shuffle=False)
    trainloaders = []
    valloaders = []
    for client_name in FederatedDataset.clients:
        trainset = load_data(
            [client_name],
            train_test_split=config.train_test_split,
            dataset_name=dataset_name.lower(),
            type="train",
            min_no_samples=config.min_num_samples,
            is_embedded=config.is_embedded)
        trainloaders.append(DataLoader(trainset, batch_size, shuffle=True))
        valset = load_data(
            [client_name],
            train_test_split=config.train_test_split,
            dataset_name=dataset_name.lower(),
            type="test",
            min_no_samples=config.min_num_samples,
            is_embedded=config.is_embedded,
        )
        valloaders.append(DataLoader(valset,batch_size, shuffle=False))

    return trainloaders, valloaders, testloader
    

