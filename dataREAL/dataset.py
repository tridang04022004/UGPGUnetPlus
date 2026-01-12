import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional

try:
    from data.joint_transforms import JointLetterbox, get_training_augmentation, get_validation_augmentation
except ImportError:
    from joint_transforms import JointLetterbox, get_training_augmentation, get_validation_augmentation


class HerlevNucleiDataset(Dataset):
    """
    Mask classes:
    - 0: background + unknown
    - 1: nuclei_small [0, 0, 255]
    - 2: nuclei_large [0, 0, 128]
    """
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
        augmentation: Optional[object] = None,
        normalize: bool = True,
        use_letterbox: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.normalize = normalize
        self.use_letterbox = use_letterbox
        
        if augmentation is None:
            if split == 'train':
                self.spatial_aug, self.image_aug = get_training_augmentation(img_size)
            else:
                self.spatial_aug, self.image_aug = get_validation_augmentation()
        else:
            if isinstance(augmentation, tuple):
                self.spatial_aug, self.image_aug = augmentation
            else:
                self.spatial_aug = augmentation
                self.image_aug = None
        
        self.split_dir = self.data_dir / split
        
        if not self.split_dir.exists():
            raise ValueError(f"Split directory not found: {self.split_dir}")
        
        self.image_files = sorted([
            f for f in self.split_dir.glob('*.BMP')
            if not f.stem.endswith('-d')
        ])
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.split_dir}")
        
        print(f"Loaded {len(self.image_files)} images from {split} split")
    
    def _rgb_to_class(self, rgb_image: np.ndarray) -> np.ndarray:

        class_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8)
        
        bg_mask = np.all(rgb_image == [255, 0, 0], axis=2)
        unknown_mask1 = np.all(rgb_image == [128, 128, 128], axis=2)
        unknown_mask2 = np.all(rgb_image == [0, 0, 0], axis=2)
        nuclei_small_mask = np.all(rgb_image == [0, 0, 255], axis=2)
        nuclei_large_mask = np.all(rgb_image == [0, 0, 128], axis=2)

        class_mask[bg_mask | unknown_mask1 | unknown_mask2] = 0
        class_mask[nuclei_small_mask] = 1
        class_mask[nuclei_large_mask] = 2
        
        return class_mask
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_files[idx]

        mask_path = image_path.parent / (image_path.stem + '-d.bmp')
        
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for image: {image_path}")

        image = Image.open(image_path).convert('RGB')

        mask = Image.open(mask_path).convert('RGB')
        mask_array = np.array(mask)
        class_mask = self._rgb_to_class(mask_array)
        mask_pil = Image.fromarray(class_mask)

        #letterbox resize
        if self.use_letterbox:
            letterbox = JointLetterbox(self.img_size, fill=0)
            image, mask_pil = letterbox([image, mask_pil])
        else:
            image = image.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            mask_pil = mask_pil.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        
        image_array = np.array(image)
        class_mask = np.array(mask_pil)
        
        # spatial augmentation 
        if self.spatial_aug is not None:
            transformed = self.spatial_aug(image=image_array, mask=class_mask)
            image_array = transformed['image']
            class_mask = transformed['mask']
        
        # image-only augmentation (color, brightness, noise)
        # NOT applied to mask
        if self.image_aug is not None:
            transformed = self.image_aug(image=image_array)
            image_array = transformed['image']

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        image_tensor = image_tensor / 255.0

        if self.normalize:
            mean = torch.tensor(self.MEAN).view(3, 1, 1)
            std = torch.tensor(self.STD).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
        
        mask_tensor = torch.from_numpy(class_mask).long()
        
        return image_tensor, mask_tensor


if __name__ == '__main__':
    data_dir = '.'
    train_dataset = HerlevNucleiDataset(data_dir, split='train')
    print(f"Training dataset size: {len(train_dataset)}")
    test_dataset = HerlevNucleiDataset(data_dir, split='test')
    print(f"Test dataset size: {len(test_dataset)}")
    image, mask = train_dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask classes: {torch.unique(mask)}")
