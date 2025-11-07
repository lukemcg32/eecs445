"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - Project 2

Challenge CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["Challenge"]


class Challenge(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        """Define model architecture."""
        super().__init__()

        # use sequential instead of having tons of forward operations
        self.features = nn.Sequential(
            
            # input: 3     output: 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),   # add more depth without changing size
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # input: 32     output: 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # input: 64     output: 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # input: 128     output: 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes) # default is 2 classes
        )

        # potentially using a single layer
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(256*4*4, num_classes),
        # )

        self.init_weights()


    
    def init_weights(self) -> None:
        """Initialize model weights."""
        torch.manual_seed(42)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

            elif isinstance(m, nn.BatchNorm2d):
                # Start as identity: mean 0, var 1
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)

        return x
