import torch
import numpy as np


def calculate_uncertainty_map(probs):
    epsilon = 1e-10
    
    if probs.dim() == 4:  
        entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=1)  
        num_classes = probs.shape[1]
    elif probs.dim() == 3:  
        entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=0)  
        num_classes = probs.shape[0]
    else:
        raise ValueError(f"Expected probs to have 3 or 4 dimensions, got {probs.dim()}")
    
    max_entropy = np.log(num_classes)
    uncertainty = entropy / max_entropy
    
    return uncertainty


def calculate_uncertainty_stats(uncertainty_map):
    return {
        'mean': uncertainty_map.mean().item(),
        'std': uncertainty_map.std().item(),
        'min': uncertainty_map.min().item(),
        'max': uncertainty_map.max().item(),
        'median': uncertainty_map.median().item()
    }
