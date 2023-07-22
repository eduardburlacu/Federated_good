import csv
import math
import os
import random
from src.Models.lstm_utils import EmbeddingTransformer, EmbeddingTransformerShakespeare
import torch
from torch.utils.data import Dataset

from torchvision import transforms
from PIL import Image

from src import PATH,PATH_src, GOD_CLIENT_NAME
#DATA_PATH, LEAF_PATH,

class FederatedDataset(Dataset):
    '''
    ---------------Info regarding data.csv
    The columns are  [file path ex 000/0000001/pt] [label ex d] [client ex George] [img ie feature X, if the dataset is embedded]
                            x[0]                      y=x[1]           cid=x[2]                     X = x[3]
    '''

    clients = None
    def __init__(self, client_name, dataset_name, transform, test_train_split, type, min_no_samples, is_embedded):
        random.seed(0)  # Ensure that test-train split is done deterministically.
        self.client_name = client_name
        self.dataset_dir = os.path.join(PATH_src['Dataset'], dataset_name)
        self.dataset_file = os.path.join(self.dataset_dir, 'data.csv')
        if FederatedDataset.clients == None:
            with open(self.dataset_file) as csvfile:
                csvreader = csv.reader(csvfile, delimiter=' ')
                rows = list(csvreader)[1:] #Eliminate head of table

            FederatedDataset.clients = {}
            for x in rows:
                if x[2] not in FederatedDataset.clients:
                    FederatedDataset.clients[x[2]] = []
                if is_embedded:
                    FederatedDataset.clients[x[2]].append((x[0], x[1], x[3]))    # Form is (path to X, y, X)
                else:
                    FederatedDataset.clients[x[2]].append((x[0], x[1]))          # Form is (path to X, y)

            for client in sorted(FederatedDataset.clients):
                random.shuffle(FederatedDataset.clients[client])

            del_clients = [c for c in FederatedDataset.clients if len(FederatedDataset.clients[c]) < min_no_samples]
            for client in del_clients:
                del FederatedDataset.clients[client]

        def get_data(client_name):
                data = FederatedDataset.clients[client_name]
                threshold = int(len(data)*test_train_split)
                if is_embedded:
                    cache = [(img, label) for _,label,img in data]
                else:
                    cache = [None] * len(data)
                if len(data) == 1 and type == "train":
                    return data, [None]
                elif len(data) == 1 and type == "test":
                    return [], []
                if type == "train":
                    return data[:threshold], cache[:threshold]
                elif type == "test":
                    return data[threshold:], cache[threshold:]
                else:
                    raise Exception(f"Unsupported type {type}.")

        if self.client_name == GOD_CLIENT_NAME:
            self.samples = [x for client in FederatedDataset.clients for x in get_data(client)[0]]
            self.cached =  [x for client in FederatedDataset.clients for x in get_data(client)[1]]
        else:
            self.samples, self.cached = get_data(client_name)
        self.transform = transform

    def __len__(self): return len(self.samples)

    # The returned value should match the format used in a model's loss() and test() functions.
    def __getitem__(self, idx):
        if self.cached[idx] is None:
            img_name, label = self.samples[idx]
            img_path = os.path.join(self.dataset_dir, img_name)
            img = torch.load(img_path)
            self.cached[idx] = (img,label)
        img,label = self.cached[idx]
        if self.transform:
            img, label = self.transform(img, label)
        return img, label

def load_data(client_names, train_test_split, dataset_name, type, min_no_samples, is_embedded):

    [client_name] = client_names

    if dataset_name == "sent140":
        transform = EmbeddingTransformer()
    elif dataset_name == "shakespeare":
        transform = EmbeddingTransformerShakespeare()
    else:
        transform = (lambda x,y: (torch.tensor(x), torch.tensor(int(float(y)))))

    dataset = FederatedDataset(
        client_name=client_name,
        dataset_name=dataset_name,
        transform=transform,
        test_train_split=train_test_split,
        type=type,
        min_no_samples=min_no_samples,
        is_embedded=is_embedded,
    )
    return dataset
