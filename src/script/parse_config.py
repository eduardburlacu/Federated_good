import toml
import os
import torchvision
from src import PATH, PATH_sim

class DatasetNode():
  def __init__(self, dataset:str):
    self.name = dataset

    if dataset=='CIFAR10':
        self.dataset = torchvision.datasets.CIFAR10
        self.task = 'vision'
        self.shape = (3,32,32)
        self.size =60000
        self.classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",)

    elif dataset=='MNIST':
        self.dataset = torchvision.datasets.MNIST
        self.task = 'vision'
        self.shape = (1,28,28)
        self.size = 60000
        self.classes = tuple(i for i in range(1,10))

    elif dataset=='FEMNIST':
        self.dataset = None
        self.task = 'vision'
        self.shape =(1,28,28)
        self.size = 805263
        self.classes = tuple("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

    else: raise ValueError('DatasetNode is broken')

    self.num_classes = len(self.classes)

  def __repr__(self): return self.name

def get_variables(file:str = 'mock'):
    '''
    Dictionary:
    ---> Federated variables
    NUM_CLIENTS = config['NUM_CLIENTS']
    NUM_ROUNDS = config['NUM_ROUNDS']
    CLIENT_FRAC = config['CLIENT_FRAC']
    MIN_FIT_CLIENTS = config['MIN_FIT_CLIENTS']
    MIN_EVALUABLE_CLIENTS = config['MIN_EVALUABLE_CLIENTS']
    MIN_AVAILABLE_CLIENTS = config['MIN_AVAILABLE_CLIENTS']

    ---> Data loading and training variables
    DATASET = config['DATASET']
    SEED = config['SEED']
    VAL_SPLIT = config['VAL_SPLIT']
    BATCH_SIZE = config['BATCH_SIZE']
    EPOCHS = config['EPOCHS']
    '''
    try: path = os.path.join(PATH['config'],file+'.toml')
    except: path = os.path.join(PATH_sim, file + '.toml')

    with open(path, 'r') as config_file:
        config_var = toml.load(config_file)
    config_var['DATASET'] = DatasetNode(config_var['DATASET'])

    return config_var
