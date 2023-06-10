import torch
from torch import nn
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.depthwise_conv = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise_conv = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class SemanticMapFeatureExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for semantic map observations."""
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim=features_dim)

        extractors = {}

        for key, subspace in observation_space.spaces.items():
            if key == "map":
                self._init_map_feature_extractor(subspace)  # type: ignore[arg-type]
                extractors[key] = self._map_feature_extractor
            elif key == "position":
                # Run through a simple MLP
                extractors[key] = nn.Sequential(
                    nn.Linear(3, 16),
                    nn.LeakyReLU(),
                    nn.Linear(16, 32),
                    nn.LeakyReLU(),
                )

        total_concat_size = self._cnn_flatten_output_dim + 32
        self.linear = nn.Sequential(nn.Linear(total_concat_size, features_dim), nn.ReLU())

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        features = torch.cat(encoded_tensor_list, dim=1)
        return self.linear(features)

   def _init_map_feature_extractor(self, map_obs_space: spaces.Box) -> None:
        n_input_channels = map_obs_space.shape[0]
        self._map_feature_extractor = nn.Sequential(
            SeparableConv3d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            SeparableConv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            SeparableConv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            SeparableConv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            SeparableConv3d(256, 512, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            self._cnn_flatten_output_dim = self._map_feature_extractor(
                torch.as_tensor(map_obs_space.sample()[None]).float()
            ).shape[1]
