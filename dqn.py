# dqn.py f
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, relu: bool = True):
        super().__init__()
        assert kernel_size in (1, 3)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=1 if kernel_size == 3 else 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.beta = nn.Parameter(torch.zeros(out_channels))
        self.relu = relu

        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        logger.info(f"ConvBlock initialized with in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}")

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x += self.beta.view(1, self.bn.num_features, 1, 1).expand_as(x)
        if self.relu:
            x = F.relu(x, inplace=True)
#        logger.debug(f"ConvBlock forward pass with input shape {x.shape}")
        return x

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, 3)
        self.conv2 = ConvBlock(channels, channels, 3, relu=False)
        logger.info(f"ResBlock initialized with channels={channels}")

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        out = F.relu(out, inplace=True)
#        logger.debug(f"ResBlock forward pass with output shape {out.shape}")
        return out

class ChessNetwork(nn.Module):
    def __init__(self, in_channels: int = 13, board_size: int = 8, residual_channels: int = 256,
                 residual_layers: int = 19):
        super().__init__()
        self.conv_input = ConvBlock(in_channels, residual_channels, 3)
        self.residual_tower = nn.Sequential(*[ResBlock(residual_channels) for _ in range(residual_layers)])

        self.policy_conv = ConvBlock(residual_channels, 2, 1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, 4096)  # 64 * 64 = 4096

        self.value_conv = ConvBlock(residual_channels, 1, 1)
        self.value_fc_1 = nn.Linear(board_size * board_size, 256)
        self.value_fc_2 = nn.Linear(256, 1)
        logger.info(f"ChessNetwork initialized with in_channels={in_channels}, board_size={board_size}, residual_channels={residual_channels}, residual_layers={residual_layers}")

    def forward(self, x):
        x = self.conv_input(x)
        x = self.residual_tower(x)

        # Policy head
        policy = self.policy_conv(x)
        policy = self.policy_fc(torch.flatten(policy, start_dim=1))

        # Value head
        value = self.value_conv(x)
        value = F.relu(self.value_fc_1(torch.flatten(value, start_dim=1)), inplace=True)
        value = torch.tanh(self.value_fc_2(value))

#        logger.debug(f"ChessNetwork forward pass completed with policy shape {policy.shape} and value shape {value.shape}")
        return policy, value
