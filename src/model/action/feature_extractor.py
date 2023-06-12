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

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512, arch: str = "3D_conv"):
        super().__init__(observation_space, features_dim=features_dim)
        self._arch = arch

        extractors = {}

        for key, subspace in observation_space.spaces.items():
            if key == "map":
                self._init_map_feature_extractor(subspace)  # type: ignore[arg-type]
                extractors[key] = self._map_feature_extractor
            elif key == "position":
                # Run through a simple MLP
                if self._arch == "3D_conv" or self._arch == "separable_conv":
                    extractors[key] = nn.Sequential(
                        nn.Linear(3, 16),
                        nn.Tanh(),
                        nn.Linear(16, 32),
                        nn.Tanh(),
                    )
                elif self._arch == "3D_conv_with_bn" or self._arch == "separable_conv_with_selu":
                    extractors[key] = nn.Sequential(
                        nn.Linear(3, 16),
                        nn.Tanh(),
                        nn.BatchNorm1d(16),
                        nn.Linear(16, 32),
                        nn.Tanh(),
                        nn.BatchNorm1d(32),
                    )
                else:
                    raise NotImplementedError

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
        if self._arch == "3D_conv":
            self._map_feature_extractor = self._3D_conv_arch(n_input_channels)
        elif self._arch == "3D_conv_with_bn":
            self._map_feature_extractor = self._3D_conv_with_bn_arch(n_input_channels)
        elif self._arch == "separable_conv":
            self._map_feature_extractor = self._seperable_conv_arch(n_input_channels)
        elif self._arch == "separable_conv_with_selu":
            self._map_feature_extractor = self._seperable_conv_with_selu_arch(n_input_channels)
        else:
            raise NotImplementedError(f"Unknown architecture: {self._arch}")

        # Compute shape by doing one forward pass
        with torch.no_grad():
            self._cnn_flatten_output_dim = self._map_feature_extractor(
                torch.as_tensor(map_obs_space.sample()[None]).float()
            ).shape[1]

    def _3D_conv_arch(self, n_input_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            nn.Conv3d(256, 512, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

    def _3D_conv_with_bn_arch(self, n_input_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            nn.Conv3d(256, 512, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

    def _seperable_conv_arch(self, n_input_channels: int) -> nn.Sequential:
        return nn.Sequential(
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

    def _seperable_conv_with_selu_arch(self, n_input_channels: int) -> nn.Sequential:
        return nn.Sequential(
            SeparableConv3d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            SeparableConv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            SeparableConv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            SeparableConv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            SeparableConv3d(256, 512, kernel_size=5, stride=1, padding=0),
            nn.SELU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # Global pooling to reduce spatial dimensions to 1x1x1
            nn.Flatten(),
        )
