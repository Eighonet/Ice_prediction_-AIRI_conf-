""" Full assembly of the parts to form the complete network """

from .unet_parts_3d import *


class UNet_3d(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_3d, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down_2d(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down_2d(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        print(x2.shape)
        x3 = self.down2(x2).squeeze(2)
        print(x3.shape)
        x4 = self.down3(x3)
        print(x4.shape)
        x5 = self.down4(x4)
        print(x5.shape)
        x = self.up1(x5, x4)
        print(x.shape)
        x = self.up2(x, x3)
        print(x.shape)
        x = self.up3(x, x2)
        print(x.shape)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
