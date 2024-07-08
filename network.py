import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Archi(nn.Module):
    def __init__(self):
        super().__init__()

        # Modify the first layer to accept single-channel input
        self.backbone1 = models.resnet18(pretrained=True)
        self.backbone1.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone1 = nn.Sequential(*list(self.backbone1.children())[:-2])

        self.block3_1 = nn.Conv2d(512, 100, 3, 1, 1)
        self.block3_2 = nn.Conv2d(100, 20, 2, 2, 1)

        self.fc = nn.Sequential(
            nn.Linear(500, 100),
            nn.GELU(),
            nn.Linear(100, 60),
            nn.GELU(),
            nn.Linear(60, 30),
            nn.GELU(),
            nn.Linear(30, 10),
            nn.GELU(),
            nn.Linear(10, 1),
        )

        self.fc3 = nn.Sequential(nn.Linear(2, 2), nn.Softmax(dim=-1))

    def forward(self, x):
        x = self.backbone1(x)
        x = F.adaptive_avg_pool2d(x, (8, 8))
        x = self.block3_1(x)
        x = self.block3_2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
