import torch
from torch import nn


# antipattern in PyTorch, don't do it!
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self._conv_block(1, 16)
        self.conv2 = self._conv_block(16, 32)
        self.conv3 = self._conv_block(32, 64)
        
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, len(classes))
        )
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x

