"""
Test-Time Augmentation (TTA) for robust predictions.

TTA applies deterministic augmentations at inference time, averages predictions,
and improves robustness without additional training.
"""
import torch
import torch.nn.functional as F


def apply_tta_transform(images, transform_type):
    """
    Apply a specific TTA transformation to images.
    
    Args:
        images: Tensor of shape (B, C, H, W)
        transform_type: One of ['original', 'hflip', 'vflip', 'rot90', 'rot180', 'rot270']
    
    Returns:
        Transformed images
    """
    if transform_type == 'original':
        return images
    elif transform_type == 'hflip':
        return torch.flip(images, dims=[3])  # Flip width
    elif transform_type == 'vflip':
        return torch.flip(images, dims=[2])  # Flip height
    elif transform_type == 'rot90':
        return torch.rot90(images, k=1, dims=[2, 3])
    elif transform_type == 'rot180':
        return torch.rot90(images, k=2, dims=[2, 3])
    elif transform_type == 'rot270':
        return torch.rot90(images, k=3, dims=[2, 3])
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")


def reverse_tta_transform(predictions, transform_type):
    """
    Reverse a TTA transformation on predictions.
    
    Args:
        predictions: Tensor of shape (B, C, H, W) - probability maps or logits
        transform_type: One of ['original', 'hflip', 'vflip', 'rot90', 'rot180', 'rot270']
    
    Returns:
        Reversed predictions
    """
    if transform_type == 'original':
        return predictions
    elif transform_type == 'hflip':
        return torch.flip(predictions, dims=[3])  # Flip back width
    elif transform_type == 'vflip':
        return torch.flip(predictions, dims=[2])  # Flip back height
    elif transform_type == 'rot90':
        return torch.rot90(predictions, k=-1, dims=[2, 3])  # Rotate -90°
    elif transform_type == 'rot180':
        return torch.rot90(predictions, k=-2, dims=[2, 3])  # Rotate -180°
    elif transform_type == 'rot270':
        return torch.rot90(predictions, k=-3, dims=[2, 3])  # Rotate -270°
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")


def predict_with_tta(model, images, device, transforms=None):
    """
    Perform Test-Time Augmentation (TTA) for robust predictions.
    
    Applies multiple deterministic augmentations, runs inference on each,
    reverses the augmentations, and averages the probability maps.
    
    Args:
        model: PyTorch model in eval mode
        images: Tensor of shape (B, C, H, W)
        device: Device to run inference on
        transforms: List of transform types to apply. 
                   Default: ['original', 'hflip', 'vflip', 'rot90', 'rot180', 'rot270']
    
    Returns:
        averaged_probs: Averaged probability maps of shape (B, C, H, W)
        predictions: Final predictions (argmax) of shape (B, H, W)
    """
    if transforms is None:
        transforms = ['original', 'hflip', 'vflip', 'rot90', 'rot180', 'rot270']
    
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for transform_type in transforms:
            # Apply augmentation
            transformed_images = apply_tta_transform(images, transform_type)
            
            # Forward pass
            outputs = model(transformed_images)
            probs = F.softmax(outputs, dim=1)
            
            # Reverse augmentation on predictions
            reversed_probs = reverse_tta_transform(probs, transform_type)
            
            all_probs.append(reversed_probs)
    
    # Average all probability maps
    averaged_probs = torch.stack(all_probs, dim=0).mean(dim=0)
    
    # Get final predictions
    predictions = torch.argmax(averaged_probs, dim=1)
    
    return averaged_probs, predictions


def get_tta_config(mode='standard'):
    """
    Get TTA configuration.
    
    Args:
        mode: 'standard' (6 transforms), 'flips_only' (3 transforms), 
              'rotations_only' (4 transforms), or 'minimal' (original only)
    
    Returns:
        List of transform types
    """
    configs = {
        'standard': ['original', 'hflip', 'vflip', 'rot90', 'rot180', 'rot270'],
        'flips_only': ['original', 'hflip', 'vflip'],
        'rotations_only': ['original', 'rot90', 'rot180', 'rot270'],
        'minimal': ['original'],
    }
    
    return configs.get(mode, configs['standard'])


if __name__ == '__main__':
    # Test TTA transforms
    print("Testing TTA transformations...")
    
    # Create dummy data
    dummy_images = torch.randn(2, 3, 128, 128)
    
    # Test each transform and its reverse
    transforms = ['original', 'hflip', 'vflip', 'rot90', 'rot180', 'rot270']
    
    for transform_type in transforms:
        # Apply transform
        transformed = apply_tta_transform(dummy_images, transform_type)
        
        # Reverse transform
        reversed_transform = reverse_tta_transform(transformed, transform_type)
        
        # Check if we get back the original
        diff = torch.abs(dummy_images - reversed_transform).max().item()
        
        print(f"{transform_type:10s}: Max difference after round-trip = {diff:.6f}")
        assert diff < 1e-6, f"Transform {transform_type} is not reversible!"
    
    print("\n✓ All TTA transforms are correctly reversible!")
    
    # Test configurations
    print("\nAvailable TTA configurations:")
    for mode in ['standard', 'flips_only', 'rotations_only', 'minimal']:
        config = get_tta_config(mode)
        print(f"  {mode:15s}: {len(config)} transforms - {config}")
