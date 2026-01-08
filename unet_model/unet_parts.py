import torch.nn.functional as F
import torch.nn as nn
import torch

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x
    
class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        # upsampling could be learned too, if provided device has enough memory
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class ResidualModule(nn.Module):
    """
    Residual Module for PGU-net+ architecture.
    
    Implements y(p) = F(X(p)) + G(X(p)) where:
    - F(x): Expansive path with multiple convolutions then upsample
    - G(x): Residual shortcut with 1x1 conv then upsample
    - Output: Element-wise sum of both paths
    
    This forces:
    - Coarse scales to learn approximate shape
    - Fine scales to learn only residuals (details)
    
    Args:
        in_ch: Number of input channels
        out_ch: Number of output channels
        bilinear: Use bilinear upsampling (True) or transposed conv (False)
    """
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(ResidualModule, self).__init__()
        
        # Path G: Residual shortcut (1x1 conv -> upsample)
        self.path_g = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),  # Channel alignment
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear 
                else nn.ConvTranspose2d(out_ch, out_ch, 2, stride=2)
        )
        
        # Path F: Expansive function (3x3 -> 3x3 -> 1x1 -> upsample)
        self.path_f_convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
        self.path_f_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear \
            else nn.ConvTranspose2d(out_ch, out_ch, 2, stride=2)
    
    def forward(self, x):
        """
        Args:
            x: Input from previous decoder layer (lower resolution)
            
        Returns:
            Element-wise sum of expansive path and residual path
        """
        # Path G: Direct residual connection
        residual = self.path_g(x)
        
        # Path F: Expansive processing
        expansive = self.path_f_convs(x)
        expansive = self.path_f_upsample(expansive)
        
        # Element-wise addition
        output = expansive + residual
        
        return output


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x