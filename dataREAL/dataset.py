import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional

try:
    from data.joint_transforms import JointLetterbox
except ImportError:
    from joint_transforms import JointLetterbox


class HerlevNucleiDataset(Dataset):
    """
    Dataset class for loading Herlev nuclei images and masks.
    
    Images and masks come in pairs:
    - name.BMP: original image
    - name-d.bmp: RGB mask
    
    Mask classes:
    - 0: background + unknown
    - 1: nuclei_small [0, 0, 255]
    - 2: nuclei_large [0, 0, 128]
    """
    
    # RGB values for each class
    MASK_COLORS = {
        'background': [255, 0, 0],
        'unknown_1': [128, 128, 128],
        'unknown_2': [0, 0, 0],
        'nuclei_small': [0, 0, 255],
        'nuclei_large': [0, 0, 128],
    }
    
    # Dataset normalization statistics (calculated from training set)
    MEAN = [0.001629576669074595, 0.001527055399492383, 0.0018455138197168708]
    STD = [0.0010540001094341278, 0.000999815878458321, 0.0011158603010699153]
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        img_size: Tuple[int, int] = (256, 256),
        joint_transform: Optional[object] = None,
        transform: Optional[object] = None,
        mask_transform: Optional[object] = None,
        normalize: bool = True,
        use_letterbox: bool = True,
    ):
        """
        Args:
            data_dir: Path to the data directory containing train/test folders
            split: Either 'train' or 'test'
            img_size: Target image size as (height, width). Default (256, 256)
            joint_transform: Transforms applied to both image and mask together (e.g., random crop, flip)
            transform: Optional transforms to apply to images only
            mask_transform: Optional transforms to apply to masks only
            normalize: Whether to apply normalization with mean/std. Default True
            use_letterbox: Whether to use aspect-ratio preserving resize with padding (recommended). Default True
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.joint_transform = joint_transform
        self.transform = transform
        self.mask_transform = mask_transform
        self.normalize = normalize
        self.use_letterbox = use_letterbox
        
        # Get the split directory
        self.split_dir = self.data_dir / split
        
        if not self.split_dir.exists():
            raise ValueError(f"Split directory not found: {self.split_dir}")
        
        # Find all image files (*.BMP), excluding mask files (*-d.bmp)
        self.image_files = sorted([
            f for f in self.split_dir.glob('*.BMP')
            if not f.stem.endswith('-d')
        ])
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.split_dir}")
        
        print(f"Loaded {len(self.image_files)} images from {split} split")
    
    def _rgb_to_class(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Convert RGB mask to class labels.
        
        Args:
            rgb_image: RGB image array of shape (H, W, 3)
            
        Returns:
            Class label array of shape (H, W) with values 0, 1, or 2
        """
        # Initialize class mask
        class_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8)
        
        # Create masks for each color
        bg_mask = np.all(rgb_image == [255, 0, 0], axis=2)
        unknown_mask1 = np.all(rgb_image == [128, 128, 128], axis=2)
        unknown_mask2 = np.all(rgb_image == [0, 0, 0], axis=2)
        nuclei_small_mask = np.all(rgb_image == [0, 0, 255], axis=2)
        nuclei_large_mask = np.all(rgb_image == [0, 0, 128], axis=2)
        
        # Assign classes
        # Class 0: background and unknown
        class_mask[bg_mask | unknown_mask1 | unknown_mask2] = 0
        # Class 1: nuclei_small
        class_mask[nuclei_small_mask] = 1
        # Class 2: nuclei_large
        class_mask[nuclei_large_mask] = 2
        
        return class_mask
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get image and mask pair.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, mask) as torch tensors
        """
        # Get image file path
        image_path = self.image_files[idx]
        
        # Construct mask file path (same name but with -d.bmp)
        mask_path = image_path.parent / (image_path.stem + '-d.bmp')
        
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for image: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Load mask and convert to class labels
        mask = Image.open(mask_path).convert('RGB')
        mask_array = np.array(mask)
        class_mask = self._rgb_to_class(mask_array)
        mask_pil = Image.fromarray(class_mask)
        
        # Apply letterbox resize (aspect-ratio preserving) if enabled
        if self.use_letterbox:
            letterbox = JointLetterbox(self.img_size, fill=0)
            image, mask_pil = letterbox([image, mask_pil])
        else:
            # Fallback to direct resize (not recommended for medical segmentation)
            image = image.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            mask_pil = mask_pil.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        
        # Apply joint transforms (e.g., random crop, flip) to both image and mask
        if self.joint_transform is not None:
            image, mask_pil = self.joint_transform([image, mask_pil])
        
        # Convert to numpy arrays
        image_array = np.array(image)
        class_mask = np.array(mask_pil)
        
        # Apply separate transforms if provided
        if self.transform is not None:
            image_array = self.transform(image=image_array)['image']
        
        if self.mask_transform is not None:
            class_mask = self.mask_transform(image=class_mask)['image']
        
        # Convert to torch tensors
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        
        # Normalize image to [0, 1] range
        image_tensor = image_tensor / 255.0
        
        # Apply normalization if requested
        if self.normalize:
            mean = torch.tensor(self.MEAN).view(3, 1, 1)
            std = torch.tensor(self.STD).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
        
        mask_tensor = torch.from_numpy(class_mask).long()
        
        return image_tensor, mask_tensor


if __name__ == '__main__':
    # Example usage
    data_dir = '.'
    
    # Load training dataset
    train_dataset = HerlevNucleiDataset(data_dir, split='train')
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Load test dataset
    test_dataset = HerlevNucleiDataset(data_dir, split='test')
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Get a sample
    image, mask = train_dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask classes: {torch.unique(mask)}")
