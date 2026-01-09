"""
Boundary Loss for improved segmentation at object boundaries.

Based on distance transform weighting - pixels closer to boundaries 
receive higher loss weights to improve boundary precision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    def __init__(self, theta=5.0):
        """
        Boundary Loss using distance transform weighting.
        
        Args:
            theta (float): Controls weight decay from boundary.
                          Lower theta = sharper focus on boundaries
                          Higher theta = softer, wider boundary region
                          Default: 5.0 (balanced)
        """
        super(BoundaryLoss, self).__init__()
        self.theta = theta
    
    def forward(self, inputs, targets):
        """
        Compute boundary-weighted loss.
        
        Args:
            inputs: Model logits (B, C, H, W)
            targets: Ground truth masks (B, H, W) with class indices
        
        Returns:
            Boundary-weighted cross-entropy loss
        """
        # Step 1: Extract boundaries from ground truth
        boundaries = self.get_boundaries(targets)  # (B, H, W)
        
        # Step 2: Compute distance transform
        distance_map = self.compute_distance_map(boundaries)  # (B, H, W)
        
        # Step 3: Convert to weights (closer to boundary = higher weight)
        # Weight = exp(-distance / theta)
        boundary_weights = torch.exp(-distance_map / self.theta)
        
        # Step 4: Compute weighted cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # (B, H, W)
        
        # Apply boundary weights
        weighted_loss = boundary_weights * ce_loss
        
        return weighted_loss.mean()
    
    def get_boundaries(self, masks):
        """
        Extract boundaries using morphological operations.
        Boundary = Dilation - Erosion
        
        Args:
            masks: Ground truth masks (B, H, W)
        
        Returns:
            Binary boundary maps (B, H, W)
        """
        boundaries = []
        
        for mask in masks:
            # Convert to one-hot for each class
            num_classes = int(mask.max().item()) + 1
            mask_one_hot = F.one_hot(mask.long(), num_classes=num_classes)  # (H, W, C)
            mask_one_hot = mask_one_hot.permute(2, 0, 1).float()  # (C, H, W)
            
            class_boundaries = []
            for c in range(num_classes):
                class_mask = mask_one_hot[c].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                
                # Dilation (max pooling)
                dilated = F.max_pool2d(class_mask, kernel_size=3, stride=1, padding=1)
                
                # Erosion (min pooling = -max_pool(-x))
                eroded = -F.max_pool2d(-class_mask, kernel_size=3, stride=1, padding=1)
                
                # Boundary = dilated - eroded (pixels that changed)
                boundary = (dilated - eroded).squeeze()
                class_boundaries.append(boundary)
            
            # Combine boundaries from all classes
            combined_boundary = torch.stack(class_boundaries).sum(dim=0)
            boundaries.append(combined_boundary > 0)
        
        return torch.stack(boundaries).float()
    
    def compute_distance_map(self, boundaries):
        """
        Compute approximate distance transform from boundaries.
        Uses iterative dilation to propagate distance values.
        
        Args:
            boundaries: Binary boundary maps (B, H, W)
        
        Returns:
            Distance maps (B, H, W) - distance from nearest boundary pixel
        """
        distance_maps = []
        max_iterations = 15  # Maximum distance to compute (in pixels)
        
        for boundary in boundaries:
            # Initialize distance map
            dist_map = torch.zeros_like(boundary)
            
            # Pixels on boundary have distance 0
            current_region = boundary.clone()
            
            # Iteratively grow regions and assign distances
            for distance in range(1, max_iterations + 1):
                # Dilate current region (grow by 1 pixel)
                dilated = F.max_pool2d(
                    current_region.unsqueeze(0).unsqueeze(0),
                    kernel_size=3,
                    stride=1,
                    padding=1
                ).squeeze()
                
                # Find newly covered pixels
                new_pixels = (dilated > current_region).float()
                
                # Assign distance value to new pixels
                dist_map = dist_map + new_pixels * distance
                
                # Update current region
                current_region = dilated
                
                # Stop if no new pixels were added
                if new_pixels.sum() == 0:
                    break
            
            distance_maps.append(dist_map)
        
        return torch.stack(distance_maps)


class EdgeWeightedLoss(nn.Module):
    """
    Alternative simpler boundary loss using edge detection.
    Applies higher weights to edge pixels.
    """
    def __init__(self, edge_weight=10.0):
        """
        Args:
            edge_weight (float): Weight multiplier for edge pixels
                                Default: 10.0
        """
        super(EdgeWeightedLoss, self).__init__()
        self.edge_weight = edge_weight
        
        # Sobel filters for edge detection
        self.register_buffer('sobel_x', torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))
    
    def forward(self, inputs, targets):
        """
        Compute edge-weighted loss.
        
        Args:
            inputs: Model logits (B, C, H, W)
            targets: Ground truth masks (B, H, W)
        
        Returns:
            Edge-weighted cross-entropy loss
        """
        # Extract edges from ground truth
        edge_mask = self.extract_edges(targets)  # (B, H, W)
        
        # Standard cross-entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # (B, H, W)
        
        # Create weight map: higher weight for edge pixels
        weights = torch.ones_like(edge_mask)
        weights = weights + edge_mask * (self.edge_weight - 1.0)
        
        # Apply weights
        weighted_loss = weights * ce_loss
        
        return weighted_loss.mean()
    
    def extract_edges(self, masks):
        """
        Extract edges using Sobel filters.
        
        Args:
            masks: Ground truth masks (B, H, W)
        
        Returns:
            Binary edge maps (B, H, W)
        """
        edges = []
        
        for mask in masks:
            mask_float = mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            
            # Apply Sobel filters
            grad_x = F.conv2d(mask_float, self.sobel_x, padding=1)
            grad_y = F.conv2d(mask_float, self.sobel_y, padding=1)
            
            # Gradient magnitude
            gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
            
            # Threshold to get binary edge map
            edge = (gradient_magnitude > 0.1).float().squeeze()
            edges.append(edge)
        
        return torch.stack(edges)
