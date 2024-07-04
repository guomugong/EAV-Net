import torch.nn.functional as F
from .unet_parts import *
from model.backbone import backbone


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder 
        self.backbone = backbone()

        # Decoder -- AV Branch
        self.up11 = Up(1024, 256, bilinear)
        self.up22 = Up(512, 128, bilinear)
        self.up33 = Up(256, 64, bilinear)
        self.up44 = Up(128, 64, bilinear)
        self.out_av = OutConv(64, n_classes)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.backbone(x)

        av1 = self.up11(x5, x4)
        av2 = self.up22(av1, x3)
        av3 = self.up33(av2, x2)
        av4 = self.up44(av3, x1)

        out_av = self.out_av(av4)
        return out_av, x3, x4, x5
