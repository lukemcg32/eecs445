"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - Project 2

Target CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.target import Target
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["Target"]


class Target(nn.Module):
    def __init__(self) -> None:
        """Define model architecture."""
        super().__init__()

        # TODO: 2(b) - define each layer

        self.conv1 = torch.nn.Conv2d(3, 16, 5, 2, padding='same')
        self.pool = torch.nn.MaxPool2d(2, stride=2)
        self.conv2 = torch.nn.Conv2d(16, 64, 5, 2, padding='same')
        self.conv3 = torch.nn.Conv2d(64, 8, 5, 2, padding='same')
        self.fc_1 = torch.nn.Linear(32, 2)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize model weights."""
        torch.manual_seed(42)
        for conv in [self.conv1, self.conv2, self.conv3]:
            # TODO: 2(b) - initialize the parameters for the convolutional layers
            v = (1.0 / (5 * 5 * conv.in_channels)) ** 0.5
            torch.nn.init.normal_(conv.weight, mean=0.0, std=v)

        # TODO: 2(b) - initialize the parameters for [self.fc_1]
        v = (1.0 / self.fc_1.in_features) ** 0.5
        torch.nn.init.normal_(self.fc_1.weight, mean=0.0, std=v)
        torch.nn.init.constant_(self.fc_1.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape

        # TODO: 2(b) - , forward pass
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))

        # flatten before putting through fully connected layer
        x = torch.flatten(x)
        x = self.fc_1(x)
