import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, img_size=(256, 256)):
        super().__init__()
        self.img_size = img_size
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)