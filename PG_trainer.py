import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from math import ceil
from PIL import Image
import torchvision.transforms as transforms

from data.dataset import HerlevNucleiDataset
from unet_model.unet import UNet1, UNet2, UNet3, UNet4
from unet_model.dice_loss import dice_coeff


def calculate_f1_score(pred, target):
    """Calculate F1 score"""
    tp = torch.sum((pred == 1) & (target == 1)).float()
    fp = torch.sum((pred == 1) & (target == 0)).float()
    fn = torch.sum((pred == 0) & (target == 1)).float()
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    return f1.item(), precision.item(), recall.item()


def calculate_iou(pred, target):
    """Calculate IoU (Jaccard Index)"""
    intersection = torch.sum((pred == 1) & (target == 1)).float()
    union = torch.sum((pred == 1) | (target == 1)).float()
    
    iou = intersection / (union + 1e-7)
    return iou.item()


def calculate_dice_score(pred, target, num_classes=3):
    """Calculate Dice score for all classes"""
    dice_scores = []
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        dice = dice_coeff(pred_c, target_c).item()
        dice_scores.append(dice)
    return np.mean(dice_scores)


def transfer_weights(old_state_dict, new_state_dict):
    """
    Transfer weights from smaller model to larger model.
    Only copies matching layer names and shapes.
    """
    transferred_dict = new_state_dict.copy()
    
    for key in old_state_dict.keys():
        if key in new_state_dict:
            old_shape = old_state_dict[key].shape
            new_shape = new_state_dict[key].shape
            
            # Only transfer if shapes match
            if old_shape == new_shape:
                transferred_dict[key] = old_state_dict[key]
                print(f"  Transferred: {key} {old_shape}")
            else:
                print(f"  Skipped (shape mismatch): {key} {old_shape} -> {new_shape}")
        else:
            print(f"  Skipped (not in new model): {key}")
    
    return transferred_dict


class ProgressiveTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize wandb
        if args.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name if args.wandb_run_name else "PGU-Net",
                config={
                    'batch_size': args.batch_size,
                    'learning_rate': args.lr,
                    'epochs': args.epochs,
                    'model': 'PGU-Net',
                    'dataset': 'Herlev Nuclei',
                    'device': str(self.device),
                    'letterboxing': True,
                    'stage_epochs': args.stage_epochs,
                    'optimizer': 'RMSprop',
                }
            )
            print("Wandb initialized successfully")
        
        # Create output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Progressive training state
        self.current_stage = 1
        self.stage_img_sizes = [32, 64, 128, 256]
        self.stage_models = [UNet1, UNet2, UNet3, UNet4]
        self.stage_epochs = args.stage_epochs  # Epochs per stage
        
        # Initialize with first stage
        self.img_size = self.stage_img_sizes[0]
        self.load_datasets()
        
        # Initialize model
        print(f"\n=== Stage 1: Starting with UNet1 @ {self.img_size}x{self.img_size} ===")
        self.model = UNet1(n_channels=3, n_classes=3).to(self.device)
        
        # Optimizer and loss
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.RMSprop(
            self.model.parameters(), 
            lr=args.lr, 
            weight_decay=1e-4
        )
        
        # Tracking metrics
        self.best_loss = float('inf')
        self.best_dice = 0.0
        self.history = {
            'train_loss': [], 'train_f1': [], 'train_dice': [], 'train_iou': [],
            'test_loss': [], 'test_f1': [], 'test_dice': [], 'test_iou': [],
            'epoch': [], 'stage': [], 'img_size': []
        }
        
        # Load checkpoint if provided
        self.start_epoch = 0
        if args.checkpoint:
            self.load_checkpoint(args.checkpoint)
    
    def load_datasets(self):
        """Load datasets with current image size"""
        print(f"Loading datasets with image size {self.img_size}x{self.img_size}...")
        
        self.train_dataset = HerlevNucleiDataset(
            self.args.data_dir, 
            split='train', 
            img_size=(self.img_size, self.img_size)
        )
        self.test_dataset = HerlevNucleiDataset(
            self.args.data_dir, 
            split='test', 
            img_size=(self.img_size, self.img_size)
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        print(f"Train: {len(self.train_dataset)} | Test: {len(self.test_dataset)}")
    
    def upgrade_model(self, new_stage):
        """Upgrade to next stage model with weight transfer"""
        print(f"\n{'='*60}")
        print(f"=== Stage {new_stage}: Upgrading to UNet{new_stage} @ {self.stage_img_sizes[new_stage-1]}x{self.stage_img_sizes[new_stage-1]} ===")
        print(f"{'='*60}\n")
        
        # Save current model state
        old_state_dict = self.model.state_dict()
        
        # Create new model
        self.img_size = self.stage_img_sizes[new_stage - 1]
        new_model = self.stage_models[new_stage - 1](n_channels=3, n_classes=3).to(self.device)
        
        # Transfer weights
        print("Transferring weights from previous stage...")
        new_state_dict = new_model.state_dict()
        transferred_state_dict = transfer_weights(old_state_dict, new_state_dict)
        new_model.load_state_dict(transferred_state_dict)
        
        # Update model
        self.model = new_model
        
        # Reload datasets with new image size
        self.load_datasets()
        
        # Create new optimizer with different learning rates
        # New layers get higher LR, transferred layers get lower LR
        base_param_ids = []
        for key in old_state_dict.keys():
            if key in transferred_state_dict:
                if old_state_dict[key].shape == transferred_state_dict[key].shape:
                    # This parameter was transferred
                    for name, param in self.model.named_parameters():
                        if key.replace('module.', '') == name:
                            base_param_ids.append(id(param))
        
        new_params = []
        base_params = []
        for param in self.model.parameters():
            if id(param) in base_param_ids:
                base_params.append(param)
            else:
                new_params.append(param)
        
        # Higher LR for new layers, lower for transferred
        if new_stage < 4:
            lr_new = self.args.lr
            lr_base = self.args.lr * 0.01  # 100x smaller for transferred weights
        else:
            # Final stage: use same LR for all
            lr_new = self.args.lr
            lr_base = self.args.lr
        
        print(f"New layers LR: {lr_new:.2e} | Transferred layers LR: {lr_base:.2e}")
        
        self.optimizer = optim.RMSprop([
            {'params': new_params, 'lr': lr_new},
            {'params': base_params, 'lr': lr_base}
        ], weight_decay=1e-4)
        
        self.current_stage = new_stage
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_f1 = 0.0
        total_iou = 0.0
        all_preds = []
        all_targets = []
        
        with tqdm(self.train_loader, desc=f'Training (Stage {self.current_stage})') as pbar:
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Calculate metrics
                total_loss += loss.item()
                
                with torch.no_grad():
                    probs = torch.exp(outputs)
                    preds = torch.argmax(probs, dim=1)
                    
                    # Store for dice calculation
                    all_preds.append(preds.cpu())
                    all_targets.append(masks.cpu())
                    
                    # F1 and IoU for nuclei classes
                    for c in range(1, 3):
                        pred_class = (preds == c).float()
                        target_class = (masks == c).float()
                        
                        f1, _, _ = calculate_f1_score(pred_class, target_class)
                        iou = calculate_iou(pred_class, target_class)
                        
                        total_f1 += f1
                        total_iou += iou
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate dice score
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        dice_score = calculate_dice_score(all_preds, all_targets)
        
        avg_loss = total_loss / len(self.train_loader)
        avg_f1 = total_f1 / (len(self.train_loader) * 2)
        avg_iou = total_iou / (len(self.train_loader) * 2)
        
        return avg_loss, avg_f1, dice_score, avg_iou
    
    def log_predictions_to_wandb(self, num_samples=8):
        """Log segmentation predictions to wandb"""
        if not self.args.use_wandb:
            return
        
        self.model.eval()
        logged_samples = 0
        wandb_images = []
        
        with torch.no_grad():
            for images, masks in self.test_loader:
                if logged_samples >= num_samples:
                    break
                
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Get predictions
                outputs = self.model(images)
                probs = torch.exp(outputs)
                preds = torch.argmax(probs, dim=1)
                
                # Log each image in the batch
                batch_size = images.size(0)
                for i in range(min(batch_size, num_samples - logged_samples)):
                    # Convert image from tensor to numpy (H, W, C)
                    img_np = images[i].cpu().permute(1, 2, 0).numpy()
                    # Denormalize if needed (assuming images are in [0, 1] range)
                    img_np = (img_np * 255).astype(np.uint8)
                    
                    # Get masks
                    gt_mask = masks[i].cpu().numpy()
                    pred_mask = preds[i].cpu().numpy()
                    
                    # Create wandb Image with masks
                    wandb_image = wandb.Image(
                        img_np,
                        masks={
                            "ground_truth": {
                                "mask_data": gt_mask,
                                "class_labels": {
                                    0: "background",
                                    1: "nucleus_type1",
                                    2: "nucleus_type2"
                                }
                            },
                            "prediction": {
                                "mask_data": pred_mask,
                                "class_labels": {
                                    0: "background",
                                    1: "nucleus_type1",
                                    2: "nucleus_type2"
                                }
                            }
                        }
                    )
                    wandb_images.append(wandb_image)
                    logged_samples += 1
                
                if logged_samples >= num_samples:
                    break
        
        # Log all images
        wandb.log({
            f"Segmentation_Outputs/Stage_{self.current_stage}": wandb_images
        })
        print(f"Logged {logged_samples} segmentation predictions to wandb")
    
    def evaluate(self):
        """Evaluate on test set"""
        self.model.eval()
        total_loss = 0.0
        total_f1 = 0.0
        total_iou = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            with tqdm(self.test_loader, desc=f'Evaluating (Stage {self.current_stage})') as pbar:
                for images, masks in pbar:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    
                    total_loss += loss.item()
                    
                    probs = torch.exp(outputs)
                    preds = torch.argmax(probs, dim=1)
                    
                    # Store for dice calculation
                    all_preds.append(preds.cpu())
                    all_targets.append(masks.cpu())
                    
                    for c in range(1, 3):
                        pred_class = (preds == c).float()
                        target_class = (masks == c).float()
                        
                        f1, _, _ = calculate_f1_score(pred_class, target_class)
                        iou = calculate_iou(pred_class, target_class)
                        
                        total_f1 += f1
                        total_iou += iou
        
        # Calculate dice score
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        dice_score = calculate_dice_score(all_preds, all_targets)
        
        avg_loss = total_loss / len(self.test_loader)
        avg_f1 = total_f1 / (len(self.test_loader) * 2)
        avg_iou = total_iou / (len(self.test_loader) * 2)
        
        return avg_loss, avg_f1, dice_score, avg_iou
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'stage': self.current_stage,
            'img_size': self.img_size,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_dice': self.best_dice,
        }
        
        # Save latest checkpoint
        latest_path = self.output_dir / 'latest_checkpoint_PG.pt'
        torch.save(checkpoint, latest_path)
        
        # Save stage checkpoint
        stage_path = self.output_dir / f'stage{self.current_stage}_checkpoint.pt'
        torch.save(checkpoint, stage_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'best_model_PG.pt'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (Dice: {self.best_dice:.4f}) to {best_path}")
            
            # Log to wandb
            if self.args.use_wandb:
                # Save model file to wandb
                wandb.save(str(best_path))
                
                # Also log as artifact for versioning
                artifact = wandb.Artifact(
                    name=f"pgu-net-{wandb.run.id}",
                    type="model",
                    description=f"Best PGU-Net model at epoch {epoch}, stage {self.current_stage}"
                )
                artifact.add_file(str(best_path))
                wandb.log_artifact(artifact)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.current_stage = checkpoint.get('stage', 1)
        self.img_size = checkpoint.get('img_size', 32)
        
        # Load appropriate model
        model_class = self.stage_models[self.current_stage - 1]
        self.model = model_class(n_channels=3, n_classes=3).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.history = checkpoint.get('history', self.history)
        self.best_dice = checkpoint.get('best_dice', 0.0)
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {self.start_epoch}, stage {self.current_stage}")
    
    def train(self):
        """Main training loop with progressive growing"""
        print(f"\n{'='*60}")
        print(f"Starting Progressive Training for {self.args.epochs} epochs")
        print(f"Stage duration: {self.stage_epochs} epochs")
        print(f"Total samples - Train: {len(self.train_dataset)}, Test: {len(self.test_dataset)}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # Determine stage based on epoch
            target_stage = min(ceil((epoch + 1) / self.stage_epochs), 4)
            
            # Upgrade model if needed
            if target_stage > self.current_stage:
                self.upgrade_model(target_stage)
                # Log predictions at the start of new stage
                self.log_predictions_to_wandb(num_samples=8)
            
            print(f"\n--- Epoch {epoch + 1}/{self.args.epochs} | Stage {self.current_stage} | Size {self.img_size}x{self.img_size} ---")
            
            # Train
            train_loss, train_f1, train_dice, train_iou = self.train_epoch()
            print(f"Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
            
            # Evaluate
            test_loss, test_f1, test_dice, test_iou = self.evaluate()
            print(f"Test  - Loss: {test_loss:.4f}, F1: {test_f1:.4f}, Dice: {test_dice:.4f}, IoU: {test_iou:.4f}")
            
            # Update history
            self.history['epoch'].append(epoch + 1)
            self.history['stage'].append(self.current_stage)
            self.history['img_size'].append(self.img_size)
            self.history['train_loss'].append(train_loss)
            self.history['train_f1'].append(train_f1)
            self.history['train_dice'].append(train_dice)
            self.history['train_iou'].append(train_iou)
            self.history['test_loss'].append(test_loss)
            self.history['test_f1'].append(test_f1)
            self.history['test_dice'].append(test_dice)
            self.history['test_iou'].append(test_iou)
            
            # Log to wandb
            if self.args.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'stage': self.current_stage,
                    'img_size': self.img_size,
                    'train/loss': train_loss,
                    'train/f1': train_f1,
                    'train/dice': train_dice,
                    'train/iou': train_iou,
                    'test/loss': test_loss,
                    'test/f1': test_f1,
                    'test/dice': test_dice,
                    'test/iou': test_iou,
                })
            
            # Save checkpoint
            is_best = test_dice > self.best_dice
            if is_best:
                self.best_dice = test_dice
            
            self.save_checkpoint(epoch, is_best=is_best)
            
            # Log predictions every 10 epochs or when achieving best score
            if (epoch + 1) % 10 == 0 or is_best:
                self.log_predictions_to_wandb(num_samples=8)
            
            # Adjust learning rate every N epochs
            if (epoch + 1) % self.args.lr_decay_every == 0:
                for param_group in self.optimizer.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] = old_lr * self.args.lr_decay
                    print(f"Learning rate adjusted: {old_lr:.2e} -> {param_group['lr']:.2e}")
        
        print("\n" + "="*60)
        print("Training completed!")
        print("="*60)
        self.print_summary()
        
        # Finish wandb run
        if self.args.use_wandb:
            wandb.run.summary['best_dice'] = self.best_dice
            wandb.run.summary['final_stage'] = self.current_stage
            wandb.finish()
            print("Wandb run finished")
    
    def print_summary(self):
        """Print training summary"""
        if len(self.history['epoch']) == 0:
            return
        
        best_idx = np.argmax(self.history['test_dice'])
        print(f"\n{'='*60}")
        print("BEST RESULTS")
        print(f"{'='*60}")
        print(f"Epoch:     {self.history['epoch'][best_idx]}")
        print(f"Stage:     {self.history['stage'][best_idx]}")
        print(f"Img Size:  {self.history['img_size'][best_idx]}x{self.history['img_size'][best_idx]}")
        print(f"Test Loss: {self.history['test_loss'][best_idx]:.4f}")
        print(f"Test Dice: {self.history['test_dice'][best_idx]:.4f}")
        print(f"Test F1:   {self.history['test_f1'][best_idx]:.4f}")
        print(f"Test IoU:  {self.history['test_iou'][best_idx]:.4f}")
        print(f"{'='*60}\n")