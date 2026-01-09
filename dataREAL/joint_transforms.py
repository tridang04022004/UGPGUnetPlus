"""
Joint transforms for applying the same spatial transformations to both images and masks.
Adapted from reference implementation.
"""
from PIL import Image, ImageOps
import random
import numbers


class JointCompose:
    """Composes several transforms together."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, imgs):
        for transform in self.transforms:
            imgs = transform(imgs)
        return imgs


class JointRandomHorizontalFlip:
    """Randomly horizontally flips the given list of PIL Images with probability 0.5"""
    
    def __call__(self, imgs):
        if random.random() < 0.5:
            return [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]
        return imgs


class JointRandomVerticalFlip:
    """Randomly vertically flips the given list of PIL Images with probability 0.5"""
    
    def __call__(self, imgs):
        if random.random() < 0.5:
            return [img.transpose(Image.FLIP_TOP_BOTTOM) for img in imgs]
        return imgs


class JointRandomRotation:
    """Randomly rotates images by 0, 90, 180, or 270 degrees"""
    
    def __call__(self, imgs):
        angle = random.choice([0, 90, 180, 270])
        if angle == 0:
            return imgs
        return [img.rotate(angle, expand=False) for img in imgs]


class JointRandomCrop:
    """Crops the given list of PIL Images at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
    
    def __call__(self, imgs):
        if self.padding > 0:
            imgs = [ImageOps.expand(img, border=self.padding, fill=0) for img in imgs]
        
        w, h = imgs[0].size
        th, tw = self.size
        if w == tw and h == th:
            return imgs
        
        if w < tw or h < th:
            # If image is smaller than crop size, return as is
            return imgs
        
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return [img.crop((x1, y1, x1 + tw, y1 + th)) for img in imgs]


class JointCenterCrop:
    """Crops the given PIL Images at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, imgs):
        w, h = imgs[0].size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return [img.crop((x1, y1, x1 + tw, y1 + th)) for img in imgs]


class JointResize:
    """Resizes the given list of PIL Images to the given size.
    size: tuple of (height, width) or int for square resize
    """
    
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, imgs):
        # First image is RGB image, second is mask
        # Use BILINEAR for image, NEAREST for mask to preserve labels
        return [
            imgs[0].resize((self.size[1], self.size[0]), Image.BILINEAR),
            imgs[1].resize((self.size[1], self.size[0]), Image.NEAREST)
        ]


class JointLetterbox:
    """Resizes images preserving aspect ratio and pads to target size (letterboxing).
    
    This is the recommended approach for medical segmentation as it:
    - Preserves nuclear morphology (no distortion)
    - Maintains aspect ratio
    - Enables batch training with uniform sizes
    
    Workflow:
    1. Resize so longest side matches target size
    2. Pad shorter side with zeros (or specified fill)
    3. Apply same transformation to mask
    
    Example: 53×129 → resize to 106×256 → pad height to 256×256
    """
    
    def __init__(self, size, fill=0, pad_mode='constant'):
        """
        Args:
            size: Target size as (height, width) or int for square
            fill: Fill value for padding (default 0 for background)
            pad_mode: 'constant', 'edge', or 'reflect'
        """
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.fill = fill
        self.pad_mode = pad_mode
    
    def __call__(self, imgs):
        target_h, target_w = self.size
        
        # Get original dimensions
        w, h = imgs[0].size
        
        # Calculate scale factor: fit longest side to target
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions preserving aspect ratio
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize all images preserving aspect ratio
        resized_imgs = []
        for i, img in enumerate(imgs):
            # Use BILINEAR for image (index 0), NEAREST for mask to preserve labels
            interpolation = Image.BILINEAR if i == 0 else Image.NEAREST
            resized_imgs.append(img.resize((new_w, new_h), interpolation))
        
        # Calculate padding to center the image
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top
        
        # Pad images
        padded_imgs = []
        for i, img in enumerate(resized_imgs):
            # For mask, always use fill=0 to ensure background class
            # For image, use specified fill value
            fill_value = 0 if i > 0 else self.fill
            
            padded = ImageOps.expand(
                img,
                border=(pad_left, pad_top, pad_right, pad_bottom),
                fill=fill_value
            )
            padded_imgs.append(padded)
        
        return padded_imgs
