import os
from src import PATH_src
from src.script.get_variables import get_variables
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset

def load_datasets(split_fn = random_split):
    '''
    Given a dataset, batch size, and validation dataset fraction, it returns 3 lists of corresponding DataLoaders for train, validation, test
    '''
    config = get_variables(sim=False)
    NUM_CLIENTS = config['NUM_CLIENTS']
    DATASET = config['DATASET'].dataset
    SHAPE = config['DATASET'].shape
    SEED = config['SEED']
    VAL_SPLIT = config['VAL_SPLIT']
    BATCH_SIZE = config['BATCH_SIZE']
    EPOCHS = config['EPOCHS']
    # Download and transform dataset (train and test)
    normalizer=transforms.Normalize([0.5]*SHAPE[0], [0.5]*SHAPE[0])
    transform = transforms.Compose(
        [transforms.ToTensor(),
         normalizer,
         ]
    )
    trainset = DATASET(os.path.join(PATH_src['Dataset'],"data"), train=True, download=True, transform=transform)
    testset  = DATASET(os.path.join(PATH_src['Dataset'],"data"), train=False,download=True, transform=transform)

    # Split training set into partitions to simulate the individual dataset
    partition_size = len(trainset) // NUM_CLIENTS
    lengths = [partition_size] * NUM_CLIENTS
    datasets = split_fn(trainset, lengths, torch.Generator().manual_seed(SEED))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []

    for ds in datasets:
        len_val = int( len(ds) * VAL_SPLIT )
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = split_fn(ds, lengths, torch.Generator().manual_seed(SEED))
        trainloaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloaders, valloaders, testloader


if __name__=='__main__':
    load_datasets()