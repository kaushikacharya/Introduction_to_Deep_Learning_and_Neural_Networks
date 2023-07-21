# ResNet: Skip connection via addition

import torch
import torch.nn as nn

class SkipConnection(nn.Module):
    def __init__(self):
        super(SkipConnection, self).__init__()
        self.conv_layer1 = nn.Conv2d(3, 6, 2, stride=2, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv_layer2 = nn.Conv2d(6, 3, 2, stride=2, padding=2)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        # WRITE YOUR CODE HERE
        h1 = self.conv_layer1(input)
        self.relu(h1)
        h2 = self.conv_layer2(h1)
        self.relu(h2)
        o = h2 + input
        return o
