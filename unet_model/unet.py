import torch.nn.functional as F
import torch.nn as nn

from .unet_parts import *

class UNet1(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet1, self).__init__()
        self.inc = inconv(n_channels, 512)
        self.down4 = down(512, 512)
        # self.up1 = up(1024, 256)  # Classic U-Net with encoder skip
        self.up1 = ResidualModule(512, 256)  # PGU-net+ Residual Module
        self.outc = outconv(256, n_classes)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down4(x1)
        # x = self.up1(x2, x1)  # Classic U-Net: concatenates encoder skip x1
        x = self.up1(x2)  # PGU-net+: only uses previous decoder layer x2
        x = self.outc(x)
        return x

class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet2, self).__init__()
        self.inc = inconv(n_channels, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        # self.up1 = up(1024, 256)  # Classic U-Net with encoder skip
        # self.up2 = up(512, 128)   # Classic U-Net with encoder skip
        self.up1 = ResidualModule(512, 256)  # PGU-net+ Residual Module
        self.up2 = ResidualModule(256, 128)  # PGU-net+ Residual Module
        self.outc1 = outconv(256, n_classes)
        self.outc2 = outconv(128, n_classes)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down3(x1)
        x3 = self.down4(x2)
        # x4 = self.up1(x3, x2)  # Classic U-Net: concatenates encoder skip x2
        # x5 = self.up2(x4, x1)  # Classic U-Net: concatenates encoder skip x1
        x4 = self.up1(x3)  # PGU-net+: only uses previous decoder layer
        x5 = self.up2(x4)  # PGU-net+: only uses previous decoder layer
        x4 = self.outc1(x4)
        x5 = self.outc2(x5)

        x4 = nn.functional.interpolate(x4, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        x = x4 + x5
        return x
    
class UNet3(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet3, self).__init__()
        self.inc = inconv(n_channels, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        # self.up1 = up(1024, 256)  # Classic U-Net with encoder skip
        # self.up2 = up(512, 128)   # Classic U-Net with encoder skip
        # self.up3 = up(256, 64)    # Classic U-Net with encoder skip
        self.up1 = ResidualModule(512, 256)  # PGU-net+ Residual Module
        self.up2 = ResidualModule(256, 128)  # PGU-net+ Residual Module
        self.up3 = ResidualModule(128, 64)   # PGU-net+ Residual Module
        self.outc1 = outconv(256, n_classes)
        self.outc2 = outconv(128, n_classes)
        self.outc3 = outconv(64, n_classes)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        # x5 = self.up1(x4, x3)  # Classic U-Net: concatenates encoder skip x3
        # x6 = self.up2(x5, x2)  # Classic U-Net: concatenates encoder skip x2
        # x7 = self.up3(x6, x1)  # Classic U-Net: concatenates encoder skip x1
        x5 = self.up1(x4)  # PGU-net+: only uses previous decoder layer
        x6 = self.up2(x5)  # PGU-net+: only uses previous decoder layer
        x7 = self.up3(x6)  # PGU-net+: only uses previous decoder layer
        x5 = self.outc1(x5)
        x6 = self.outc2(x6)
        x7 = self.outc3(x7)
		
        x5 = nn.functional.interpolate(x5, scale_factor=(4, 4), mode='bilinear', align_corners=True)
        x6 = nn.functional.interpolate(x6, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        x = x5 + x6 + x7
        return x
    
class UNet4(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet4, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        # self.up1 = up(1024, 256)  # Classic U-Net with encoder skip
        # self.up2 = up(512, 128)   # Classic U-Net with encoder skip
        # self.up3 = up(256, 64)    # Classic U-Net with encoder skip
        # self.up4 = up(128, 64)    # Classic U-Net with encoder skip
        self.up1 = ResidualModule(512, 256)  # PGU-net+ Residual Module
        self.up2 = ResidualModule(256, 128)  # PGU-net+ Residual Module
        self.up3 = ResidualModule(128, 64)   # PGU-net+ Residual Module
        self.up4 = ResidualModule(64, 64)    # PGU-net+ Residual Module
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
        # x6 = self.up1(x5, x4)  # Classic U-Net: concatenates encoder skip x4
        # x7 = self.up2(x6, x3)  # Classic U-Net: concatenates encoder skip x3
        # x8 = self.up3(x7, x2)  # Classic U-Net: concatenates encoder skip x2
        # x9 = self.up4(x8, x1)  # Classic U-Net: concatenates encoder skip x1
        x6 = self.up1(x5)  # PGU-net+: only uses previous decoder layer
        
        x7 = self.up2(x6)  # PGU-net+: only uses previous decoder layer
        
        x8 = self.up3(x7)  # PGU-net+: only uses previous decoder layer
        x9 = self.up4(x8)  # PGU-net+: only uses previous decoder layer

        x6 = self.outc1(x6)
        x7 = self.outc2(x7)
        x8 = self.outc3(x8)
        x9 = self.outc4(x9)
        x6 = nn.functional.interpolate(x6, scale_factor=(8, 8), mode='bilinear', align_corners=True)
        x7 = nn.functional.interpolate(x7, scale_factor=(4, 4), mode='bilinear', align_corners=True)
        x8 = nn.functional.interpolate(x8, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        x = x6 + x7 + x8 + x9
        return x