from PIL import Image, ImageOps
import numbers
import albumentations as A

def get_training_augmentation(img_size=(256, 256)):
    spatial_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,  
            scale_limit=0.1,     
            rotate_limit=30,     
            border_mode=0,      
            p=0.5
        ),
        
        A.OneOf([
            A.ElasticTransform(
                alpha=120,
                sigma=120 * 0.05,
                alpha_affine=120 * 0.03,
                border_mode=0,
                p=1.0
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                border_mode=0,
                p=1.0
            ),
        ], p=0.3),
        
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, border_mode=0, p=1.0),
            A.Perspective(scale=(0.05, 0.1), p=1.0),
        ], p=0.2),
        
    ], additional_targets={'mask': 'mask'})
    
    image_aug = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=1.0
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=1.0
            ),
        ], p=0.5),
        
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.3),
    ])
    
    return spatial_aug, image_aug


class JointLetterbox:
    def __init__(self, size, fill=0, pad_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.fill = fill
        self.pad_mode = pad_mode
    
    def __call__(self, imgs):
        target_h, target_w = self.size
        w, h = imgs[0].size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_imgs = []
        for i, img in enumerate(imgs):
            interpolation = Image.BILINEAR if i == 0 else Image.NEAREST
            resized_imgs.append(img.resize((new_w, new_h), interpolation))
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top
        padded_imgs = []
        for i, img in enumerate(resized_imgs):
            fill_value = 0 if i > 0 else self.fill
            
            padded = ImageOps.expand(
                img,
                border=(pad_left, pad_top, pad_right, pad_bottom),
                fill=fill_value
            )
            padded_imgs.append(padded)
        
        return padded_imgs
