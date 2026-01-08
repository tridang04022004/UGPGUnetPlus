import argparse
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from data.dataset import HerlevNucleiDataset
from unet_model.unet import UNet1, UNet2, UNet3, UNet4
from unet_model.dice_loss import dice_coeff
from unet_model.unet_parts import up, down, inconv, outconv
import torch.nn as nn


# Legacy UNet4 class for loading old checkpoints with classic up() modules
class UNet4Legacy(nn.Module):
    """Legacy UNet4 using classic up() modules instead of ResidualModule"""
    def __init__(self, n_channels, n_classes):
        super(UNet4Legacy, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)  # Classic U-Net with encoder skip
        self.up2 = up(512, 128)   # Classic U-Net with encoder skip
        self.up3 = up(256, 64)    # Classic U-Net with encoder skip
        self.up4 = up(128, 64)    # Classic U-Net with encoder skip
        self.outc1 = outconv(256, n_classes)
        self.outc2 = outconv(128, n_classes)
        self.outc3 = outconv(64, n_classes)
        self.outc4 = outconv(64, n_classes)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)  # Classic U-Net: concatenates encoder skip x4
        x7 = self.up2(x6, x3)  # Classic U-Net: concatenates encoder skip x3
        x8 = self.up3(x7, x2)  # Classic U-Net: concatenates encoder skip x2
        x9 = self.up4(x8, x1)  # Classic U-Net: concatenates encoder skip x1
        x6 = self.outc1(x6)
        x7 = self.outc2(x7)
        x8 = self.outc3(x8)
        x9 = self.outc4(x9)
        x6 = nn.functional.interpolate(x6, scale_factor=(8, 8), mode='bilinear', align_corners=True)
        x7 = nn.functional.interpolate(x7, scale_factor=(4, 4), mode='bilinear', align_corners=True)
        x8 = nn.functional.interpolate(x8, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        x = x6 + x7 + x8 + x9
        return x


def detect_model_from_checkpoint(checkpoint):
    """Detect which UNet model was used based on checkpoint structure"""
    state_dict = checkpoint['model_state_dict']
    
    # Check if this is a legacy model (has up.conv instead of ResidualModule's path_g/path_f_convs)
    has_legacy_up = any('up1.conv.conv' in key for key in state_dict.keys())
    has_residual_module = any('up1.path_g' in key or 'up1.path_f_convs' in key for key in state_dict.keys())
    
    # Check if stage info is available (from progressive training)
    if 'stage' in checkpoint:
        stage = checkpoint['stage']
        # For stage 4, check if it's legacy or new architecture
        if stage == 4 and has_legacy_up:
            return UNet4Legacy, 4
        model_map = {1: UNet1, 2: UNet2, 3: UNet3, 4: UNet4}
        return model_map[stage], stage
    
    # Otherwise, infer from architecture by checking layer presence and shapes
    has_down1 = any('down1' in key for key in state_dict.keys())
    has_up4 = any('up4' in key for key in state_dict.keys())
    has_outc4 = any('outc4' in key for key in state_dict.keys())
    
    # Check inc layer channel size to distinguish models
    inc_channels = None
    for key in state_dict.keys():
        if 'inc.conv.conv.0.weight' in key:
            inc_channels = state_dict[key].shape[0]  # Output channels
            break
    
    # Determine model based on layer presence and channel counts
    if has_down1 and has_up4 and has_outc4:
        # UNet4: has down1, up4, outc4, inc has 64 channels
        if inc_channels == 64:
            # Check if legacy or new architecture
            if has_legacy_up:
                return UNet4Legacy, 4
            return UNet4, 4
    
    if not has_down1 and not has_up4 and not has_outc4:
        # Could be UNet2 or UNet3
        has_down2 = any('down2' in key for key in state_dict.keys())
        has_up3 = any('up3' in key for key in state_dict.keys())
        has_outc3 = any('outc3' in key for key in state_dict.keys())
        
        if has_down2 and has_up3 and has_outc3:
            # UNet3: has down2, down3, down4, up1, up2, up3, inc has 128 channels
            if inc_channels == 128:
                return UNet3, 3
        else:
            # UNet2: only has down3, down4, up1, up2, inc has 256 channels
            if inc_channels == 256:
                return UNet2, 2
    
    if not has_down1 and not any('down2' in key for key in state_dict.keys()):
        # UNet1: only has down4, up1, inc has 512 channels
        if inc_channels == 512:
            return UNet1, 1
    
    # Default to UNet4Legacy if detection fails and it's a legacy checkpoint
    if has_legacy_up:
        print("Warning: Could not reliably detect model type, defaulting to UNet4Legacy")
        return UNet4Legacy, 4
    
    # Default to UNet4 if detection fails
    print("Warning: Could not reliably detect model type, defaulting to UNet4")
    return UNet4, 4


class Predictor:
    def __init__(self, checkpoint_path, device='cpu'):
        """Initialize predictor with trained model"""
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # Load checkpoint and detect model architecture
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        model_class, stage = detect_model_from_checkpoint(checkpoint)
        self.model = model_class(n_channels=3, n_classes=3)
        
        # Get image size if available
        img_size = checkpoint.get('img_size', 'unknown')
        model_type = "Legacy" if model_class == UNet4Legacy else ""
        if img_size != 'unknown':
            print(f"Detected model: {model_type}UNet{stage} @ {img_size}x{img_size}".strip())
        else:
            print(f"Detected model: {model_type}UNet{stage}".strip())
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully from {checkpoint_path}")
        
        # Class names and colors for visualization
        self.class_names = ['Background', 'Nuclei Small', 'Nuclei Large']
        # Custom colormap for visualization
        self.colors = ['#FF0000', '#0000FF', '#000080']  # Red, Blue, Dark Blue
        self.cmap = ListedColormap(self.colors)
    
    def predict(self, image_tensor):
        """Make prediction on image"""
        with torch.no_grad():
            image = image_tensor.unsqueeze(0).to(self.device)
            output = self.model(image)
            probs = F.softmax(output, dim=1)  # Convert from logits to probabilities
            pred = torch.argmax(probs, dim=1)
            return pred.squeeze(0).cpu().numpy(), probs.squeeze(0).cpu().numpy()
    
    def calculate_metrics(self, pred_mask, gt_mask):
        """Calculate metrics for prediction"""
        pred_tensor = torch.from_numpy(pred_mask).float()
        gt_tensor = torch.from_numpy(gt_mask).float()
        
        metrics = {}
        for c in range(3):
            pred_class = (pred_mask == c).astype(np.float32)
            gt_class = (gt_mask == c).astype(np.float32)
            
            dice = dice_coeff(
                torch.from_numpy(pred_class).float(),
                torch.from_numpy(gt_class).float()
            ).item()
            metrics[self.class_names[c]] = dice
        
        return metrics
    
    def visualize_results(self, images_list, output_dir='./predictions'):
        """Visualize predictions vs ground truth"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, (image, gt_mask, pred_mask, metrics) in enumerate(images_list):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'Sample {idx + 1}', fontsize=14, fontweight='bold')
            
            # Denormalize image for visualization
            # Reverse normalization: img = img * std + mean
            mean = torch.tensor(HerlevNucleiDataset.MEAN).view(3, 1, 1)
            std = torch.tensor(HerlevNucleiDataset.STD).view(3, 1, 1)
            denormalized_image = image * std + mean
            denormalized_image = torch.clamp(denormalized_image, 0, 1)  # Ensure valid range
            
            # Convert to displayable format
            display_image = (denormalized_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            axes[0].imshow(display_image)
            axes[0].set_title('Input Image\n(Letterboxed)')
            axes[0].axis('off')
            
            # Ground truth mask
            axes[1].imshow(gt_mask, cmap=self.cmap, vmin=0, vmax=2)
            axes[1].set_title('Ground Truth Mask')
            axes[1].axis('off')
            
            # Predicted mask
            axes[2].imshow(pred_mask, cmap=self.cmap, vmin=0, vmax=2)
            metrics_text = '\n'.join([f'{name}: {dice:.4f}' 
                                     for name, dice in metrics.items()])
            axes[2].set_title(f'Predicted Mask\n{metrics_text}', fontsize=10)
            axes[2].axis('off')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=self.colors[0], label=self.class_names[0]),
                Patch(facecolor=self.colors[1], label=self.class_names[1]),
                Patch(facecolor=self.colors[2], label=self.class_names[2])
            ]
            fig.legend(handles=legend_elements, loc='upper center', 
                      ncol=3, bbox_to_anchor=(0.5, -0.05))
            
            plt.tight_layout()
            
            # Save figure
            save_path = output_dir / f'prediction_{idx + 1:02d}.png'
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
            plt.close()
    
    def test_on_samples(self, data_dir, num_samples=5):
        """Test on random samples from test set"""
        dataset = HerlevNucleiDataset(data_dir, split='test')
        
        # Randomly select samples
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        indices.sort()
        
        results = []
        overall_metrics = {class_name: [] for class_name in self.class_names}
        
        print(f"\nTesting on {len(indices)} random images...\n")
        
        for i, idx in enumerate(indices):
            image, gt_mask = dataset[idx]
            
            # Make prediction
            pred_mask, probs = self.predict(image)
            
            # Calculate metrics
            metrics = self.calculate_metrics(pred_mask, gt_mask.numpy())
            
            # Store overall metrics
            for class_name, dice in metrics.items():
                overall_metrics[class_name].append(dice)
            
            # Print results
            print(f"Sample {i + 1}/{len(indices)}:")
            print(f"  Image file: {dataset.image_files[idx].name}")
            for class_name, dice in metrics.items():
                print(f"  {class_name} Dice: {dice:.4f}")
            print()
            
            # Store for visualization
            results.append((image, gt_mask.numpy(), pred_mask, metrics))
        
        return results, overall_metrics
    
    def print_summary(self, overall_metrics):
        """Print overall summary"""
        print("\n" + "="*50)
        print("OVERALL SUMMARY")
        print("="*50)
        
        for class_name, dice_scores in overall_metrics.items():
            mean_dice = np.mean(dice_scores)
            std_dice = np.std(dice_scores)
            print(f"{class_name}:")
            print(f"  Mean Dice: {mean_dice:.4f}")
            print(f"  Std Dev:   {std_dice:.4f}")
            print(f"  Scores:    {[f'{d:.4f}' for d in dice_scores]}")
        
        # Overall mean dice
        all_scores = []
        for scores in overall_metrics.values():
            all_scores.extend(scores)
        overall_dice = np.mean(all_scores)
        print(f"\nOverall Mean Dice: {overall_dice:.4f}")
        print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Test UNet4 predictions on test set')
    
    parser.add_argument('--checkpoint', type=str, default='./outputs/best_model.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of random test samples to evaluate')
    parser.add_argument('--output-dir', type=str, default='./predictions',
                        help='Output directory for visualizations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create predictor and test
    predictor = Predictor(args.checkpoint, device=device)
    results, overall_metrics = predictor.test_on_samples(args.data_dir, args.num_samples)
    
    # Visualize results
    predictor.visualize_results(results, output_dir=args.output_dir)
    
    # Print summary
    predictor.print_summary(overall_metrics)


if __name__ == '__main__':
    main()
