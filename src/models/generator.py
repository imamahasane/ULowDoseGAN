import torch
import torch.nn as nn
import torch.nn.functional as F

class UltraLightUNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, channels=[16, 32, 64]):
        super().__init__()
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        
        self.encoder1 = self._make_encoder_block(channels[0], channels[0])
        self.encoder2 = self._make_encoder_block(channels[0], channels[1], downsample=True)
        self.encoder3 = self._make_encoder_block(channels[1], channels[2], downsample=True)
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(channels[2], channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(channels[1], channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)

    def _make_encoder_block(self, in_ch, out_ch, downsample=False):
        layers = []
        if downsample:
            layers.append(nn.MaxPool2d(2))
        layers.extend([
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch),  # Depthwise Conv
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ])
        return nn.Sequential(*layers)

    def forward(self, x):
        original_size = (x.size(2), x.size(3))
        x = self.initial_conv(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        d2 = self.up2(e3)
        e2_resized = F.interpolate(e2, size=(d2.size(2), d2.size(3)), mode='bilinear', align_corners=True)
        d2 = d2 + e2_resized
        
        d1 = self.up1(d2)
        e1_resized = F.interpolate(e1, size=(d1.size(2), d1.size(3)), mode='bilinear', align_corners=True)
        d1 = d1 + e1_resized
        
        output = self.final_conv(d1)
        output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=True)
        return output