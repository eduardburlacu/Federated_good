#----------------------------External Imports----------------------------
import random
import flwr as fl
from flwr.common.typing import Scalar
import torch
from torch.utils.data import DataLoader
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
    )
else: DEVICE = torch.device('cpu')

#----Insert main project directory so that we can resolve the src imports-------
import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, src_path)
#----------------------------Internal Imports-----------------------------

from src import PATH,PATH_src
from src.federated_dataset import load_data


def print_samples(dataset, num_samples=5):
    print(f"Printing {num_samples} samples from the dataset...")
    for i in range(min(len(dataset), num_samples)):
        sample = dataset[i]
        img, label = sample
        print(f"Sample {i + 1}:")
        print("Image shape:", img.shape)  # Assuming img is a tensor representing the image
        print("Label:", label)
        print("-" * 20)

if __name__=='__main__':
    print('DATASET VALIDATION...')
    dataset = load_data(['THE_TWO_GENTLEMEN_OF_VERONA_LUCETTA'], 0.75, "shakespeare", 'train', 3, True)
    # Print a few samples from the dataset
    print_samples(dataset, num_samples=5)

    print('DATALOADER VALIDATION...')
    batch_size = 4  # Adjust the batch size as needed
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch_idx, (batch_images, batch_labels) in enumerate(data_loader):
        print(f"Batch {batch_idx + 1}:")
        print("Batch images shape:", batch_images.shape)
        print("Batch labels:", batch_labels)
        print("-" * 20)
        print("First sample in the batch:")
        print("Image shape:", batch_images[0].shape)
        print("Label:", batch_labels[0])
        print("=" * 20)
        # Stop after printing the first batch to avoid too much output
        break
