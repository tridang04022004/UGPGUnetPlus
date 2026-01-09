"""
Calculate mean and standard deviation for the Herlev dataset.
Run this script to compute normalization statistics for your training data.
"""
import torch
from torch.utils.data import DataLoader
from dataset import HerlevNucleiDataset
import numpy as np
from tqdm import tqdm


def calculate_mean_std(data_dir='.', batch_size=8, num_workers=0):
    """
    Calculate mean and std for the dataset.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size for loading
        num_workers: Number of workers for data loading (0 recommended for Windows)
    
    Returns:
        mean, std as lists
    """
    print("Calculating dataset statistics...")
    
    # Create dataset without normalization
    dataset = HerlevNucleiDataset(
        data_dir=data_dir,
        split='train',
        normalize=False  # Important: don't normalize when calculating stats
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # Initialize variables for calculating mean and std
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    print("Computing mean...")
    # First pass: calculate mean
    for images, _ in tqdm(loader, desc="Calculating mean"):
        # images shape: [B, C, H, W]
        # Normalize to [0, 1]
        images = images / 255.0
        
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        total_samples += batch_samples
    
    mean = mean / total_samples
    
    print("Computing std...")
    # Second pass: calculate std
    for images, _ in tqdm(loader, desc="Calculating std"):
        images = images / 255.0
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        std += ((images - mean.view(1, 3, 1)) ** 2).sum([0, 2])
    
    std = torch.sqrt(std / (total_samples * dataset.img_size[0] * dataset.img_size[1]))
    
    mean_list = mean.tolist()
    std_list = std.tolist()
    
    print("\n" + "="*50)
    print("Dataset Statistics:")
    print("="*50)
    print(f"Mean: {mean_list}")
    print(f"Std:  {std_list}")
    print("="*50)
    print("\nAdd these values to your dataset.py:")
    print(f"MEAN = {mean_list}")
    print(f"STD = {std_list}")
    print("="*50)
    
    return mean_list, std_list


if __name__ == '__main__':
    calculate_mean_std()
