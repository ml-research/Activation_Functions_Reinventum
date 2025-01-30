# data_loader.py

import os
import math
import torch
import logging
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

# ------------------------------------------------------------------------------------
# Data Loader for Imagenette or any other dataset
# ------------------------------------------------------------------------------------
# This module is responsible for preparing your training, validation, and test loaders.
# If you want to switch to a different dataset, you can swap out the code here
# without changing how the model or test script works.
# ------------------------------------------------------------------------------------

# get path to data 
get_path = os.getcwd()
#image_path = get_path + "/data/imagenette/imagenette2-160" #benchmark bob
data_path = get_path + "/datasets/imagenette"

def get_imagenette_dataloaders(
    dataset_path: str = data_path,
    img_size: int = 160,
    batch_size: int = 16,
    train_split_ratio: float = 0.95,
    num_workers: int = 2
):
    """
    Prepare train, validation, and test DataLoaders for the Imagenette dataset.
    
    Args:
        image_path (str): Path to the parent directory containing 'train' and 'val' subfolders.
        img_size (int): The size to which images are resized (img_size x img_size).
        batch_size (int): The batch size for all DataLoaders.
        train_split_ratio (float): The fraction of the training dataset actually used for training (the rest is validation).
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        (DataLoader, DataLoader, DataLoader):
            A tuple of (train_loader, val_loader, test_loader).
    """
    logging.info("Initializing data transformations and dataloaders...")

    # Define transformations: resize, to Tensor, normalize.
    manual_transforms_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4600, 0.4549, 0.4274],
                             std=[0.2287, 0.2226, 0.2309])
    ])

    manual_transforms_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4600, 0.4549, 0.4274],
                             std=[0.2287, 0.2226, 0.2309])
    ])

    # Paths to "train" and "val" folders
    train_data_path = "/workspaces/Act_Func_Reinventum/activations/cuda_torch/datasets/imagenette/train"  #os.path.join(dataset_path, "train")
    val_data_path = "/workspaces/Act_Func_Reinventum/activations/cuda_torch/datasets/imagenette/val" #os.path.join(dataset_path, "val")

    # Datasets
    train_dataset_full = datasets.ImageFolder(train_data_path, transform=manual_transforms_train)
    test_dataset = datasets.ImageFolder(val_data_path, transform=manual_transforms_val)

    # Split train set into (train + val)
    train_len = int(train_split_ratio * len(train_dataset_full))
    val_len = len(train_dataset_full) - train_len
    train_data, val_data = random_split(train_dataset_full, [train_len, val_len])

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    logging.info(f"DataLoaders created. Train size={len(train_data)}, Val size={len(val_data)}, Test size={len(test_dataset)}")

    return train_loader, val_loader, test_loader


# ------------------------------------------------------------------------------
# ADD YOUR CODE HERE
# ------------------------------------------------------------------------------





# ------------------------------------------------------------------------------