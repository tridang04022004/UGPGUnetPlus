import torch
import torch.nn as nn
import torch.nn.functional as F
from .boundary_loss import BoundaryLoss


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = torch.tensor([alpha], dtype=torch.float32)
        else:
            self.alpha = None
    
    def forward(self, inputs, targets):
        p = F.softmax(inputs, dim=1)
        
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)  # [N, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [N, C, H, W]
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # [N, H, W]
        
        p_t = (p * targets_one_hot).sum(dim=1)  # [N, H, W]
        
        focal_weight = (1 - p_t) ** self.gamma
        
        focal_loss = focal_weight * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            alpha_t = self.alpha[targets]  # [N, H, W]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLossWithLogits(nn.Module):
    def __init__(self, smooth=1.0, ignore_background=False):
        super(DiceLossWithLogits, self).__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background
    
    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)  # [N, C, H, W]
        
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)  # [N, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [N, C, H, W]
        
        start_class = 1 if self.ignore_background else 0
        
        dice_scores = []
        for c in range(start_class, num_classes):
            pred_c = probs[:, c, :, :]  # [N, H, W]
            target_c = targets_one_hot[:, c, :, :]  # [N, H, W]
            
            pred_flat = pred_c.contiguous().view(-1)
            target_flat = target_c.contiguous().view(-1)
            
            intersection = (pred_flat * target_flat).sum()
            union = pred_flat.sum() + target_flat.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        dice_coeff = torch.stack(dice_scores).mean()
        return 1.0 - dice_coeff


class CombinedLoss(nn.Module):
    def __init__(self, 
                 focal_weight=0.4, 
                 dice_weight=0.4,
                 boundary_weight=0.2,
                 focal_gamma=2.0,
                 focal_alpha=None,
                 dice_smooth=1.0,
                 ignore_background=False,
                 boundary_theta=5.0):
        super(CombinedLoss, self).__init__()
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        
        self.focal_loss = FocalLoss(
            alpha=focal_alpha, 
            gamma=focal_gamma, 
            reduction='mean'
        )
        
        self.dice_loss = DiceLossWithLogits(
            smooth=dice_smooth,
            ignore_background=ignore_background
        )
        
        self.boundary_loss = BoundaryLoss(
            theta=boundary_theta
        )
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        boundary = self.boundary_loss(inputs, targets)
        
        combined = (self.focal_weight * focal + 
                   self.dice_weight * dice + 
                   self.boundary_weight * boundary)
        
        return combined, focal, dice, boundary
