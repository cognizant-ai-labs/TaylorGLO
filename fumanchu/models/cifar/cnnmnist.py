import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['cnnmnist']


class CNNMNIST(nn.Module):

    def __init__(self, num_classes=10, dropRate=0.4):
        super(CNNMNIST, self).__init__()
        self.dropRate = dropRate
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fullyConnected = nn.Sequential(
            nn.Linear(64*7*7, 1024),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fullyConnected(x)
        if self.dropRate > 0:
            x = F.dropout(x, p=self.dropRate, training=self.training)
        x = self.classifier(x)
        return x


def cnnmnist(**kwargs):
    model = CNNMNIST(**kwargs)
    return model
