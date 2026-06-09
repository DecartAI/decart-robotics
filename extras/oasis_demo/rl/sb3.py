"""Stable-Baselines3 helpers for A2V training examples."""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ThreeCameraFeatureExtractor(BaseFeaturesExtractor):
    """Small CNN feature extractor for left/front/right camera observations."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 384,
    ) -> None:
        super().__init__(observation_space, features_dim)
        self.camera_names = ["left_forward", "front", "right_forward"]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(nn.Linear(32 * 3, features_dim), nn.ReLU())

    def forward(self, observations):
        encoded = []
        for name in self.camera_names:
            x = observations[name]
            if x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)
            x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
            x = F.interpolate(x, size=(128, 192), mode="bilinear", align_corners=False)
            encoded.append(self.encoder(x))
        return self.proj(torch.cat(encoded, dim=1))


def torch_device_status(module: nn.Module) -> str:
    """Return a short device summary useful for spotting CPU execution."""

    devices = {str(param.device) for param in module.parameters()}
    if not devices:
        return "no parameters"
    return ", ".join(sorted(devices))
