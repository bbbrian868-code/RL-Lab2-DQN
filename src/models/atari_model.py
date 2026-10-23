import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AtariNetDQN(nn.Module):
    """Nature DQN (Mnih et al. 2015) -- unchanged from the TA sample code."""

    def __init__(self, num_classes=4, init_weights=True):
        super(AtariNetDQN, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                        nn.ReLU(True)
                                        )
        self.classifier = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, num_classes)
                                        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = x.float() / 255.
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)


class AtariNetDuelingDQN(AtariNetDQN):
    """[EXTRA: Dueling DQN] Dueling architecture (Wang et al. 2016).

    Same convolutional trunk as the Nature net; the single FC head is replaced
    by a value stream and an advantage stream recombined as

        Q(s,a) = V(s) + ( A(s,a) - mean_a' A(s,a') )

    Subtracting the mean removes the identifiability problem: without it, any
    constant could be shifted between V and A without changing Q.
    """

    def __init__(self, num_classes=4, init_weights=True):
        super(AtariNetDuelingDQN, self).__init__(num_classes, init_weights=False)
        del self.classifier
        self.value = nn.Sequential(nn.Linear(7*7*64, 512),
                                   nn.ReLU(True),
                                   nn.Linear(512, 1))
        self.advantage = nn.Sequential(nn.Linear(7*7*64, 512),
                                       nn.ReLU(True),
                                       nn.Linear(512, num_classes))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = x.float() / 255.
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        v = self.value(x)                                   # (B, 1)
        a = self.advantage(x)                               # (B, num_actions)
        return v + (a - a.mean(dim=1, keepdim=True))


def build_network(variant, num_actions):
    """`variant` is one of dqn / ddqn / dueling / dueling_ddqn."""
    if variant in ("dueling", "dueling_ddqn"):
        return AtariNetDuelingDQN(num_actions)
    return AtariNetDQN(num_actions)
