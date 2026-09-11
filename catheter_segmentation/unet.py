import torch
import torch.nn as nn
import torch.optim as optim

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNet3D(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()
        b = base_channels

        # Encoder
        self.enc1 = ConvBlock3D(1, b)
        self.enc2 = ConvBlock3D(b, b*2)
        self.enc3 = ConvBlock3D(b*2, b*4)
        self.enc4 = ConvBlock3D(b*4, b*8)

        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = ConvBlock3D(b*8, b*16)

        # Decoder
        self.up4 = nn.ConvTranspose3d(b*16, b*8, 2, stride=2)
        self.dec4 = ConvBlock3D(b*16, b*8)

        self.up3 = nn.ConvTranspose3d(b*8, b*4, 2, stride=2)
        self.dec3 = ConvBlock3D(b*8, b*4)

        self.up2 = nn.ConvTranspose3d(b*4, b*2, 2, stride=2)
        self.dec2 = ConvBlock3D(b*4, b*2)

        self.up1 = nn.ConvTranspose3d(b*2, b, 2, stride=2)
        self.dec1 = ConvBlock3D(b*2, b)

        self.tail = nn.Conv3d(b, 1, 1)

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        # Bottleneck
        x = self.bottleneck(self.pool(s4))

        # Decoder
        x = self.dec4(torch.cat([self.up4(x), s4], dim=1))
        x = self.dec3(torch.cat([self.up3(x), s3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1))

        return self.tail(x)
