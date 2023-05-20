from typing import Generator, Tuple

import numpy as np
import torch

from ...utils import datatypes


class Counter:
    """Counter that increments from 0 to max-1 at each step, and then resets to 0
    Used for polling the global policy at a fixed frequency

    Args:
        max (int): The maximum value of the counter
    """

    def __init__(self, max: int):
        self._max = max
        self._counter = 0

    def step(self) -> None:
        """Advance the counter by 1. If the counter reaches max, it is reset to 0"""
        self._counter = (self._counter + 1) % self._max

    def is_zero(self) -> bool:
        """Return True if the counter is 0, False otherwise"""
        return self._counter == 0


class ObservationCache:
    """Cache for observations.
    This is used so that the perception model can be called with a batch of observations, before
    the global policy is called. This is more efficient than calling the perception model for each
    observation individually.
    """

    def __init__(self):
        self._rgb = []
        self._depth = []
        self._poses = []

    def add(self, rgb: datatypes.RGBImage, depth: datatypes.DepthMap, pose: datatypes.Pose) -> None:
        """Add an observation to the cache. The cache is FIFO

        Args:
            rgb (datatypes.RGBImage): RGB image. in range [0, 255]
            depth (datatypes.DepthMap): Depth map
            pose (datatypes.Pose): Pose (position, rotation)
        """
        self._rgb.append(rgb / 255)
        self._depth.append(depth)
        self._poses.append(pose)

    def clear(self) -> None:
        """Clear the cache"""
        self._rgb.clear()
        self._depth.clear()
        self._poses.clear()

    def get(
        self,
    ) -> Generator[Tuple[datatypes.RGBImage, datatypes.DepthMap, datatypes.Pose], None, None]:
        """Generator that yields the observations in the cache (FIFO)

        Yields:
            datatype.RGBImage: RGB image in range [0, 1]
            datatype.DepthMap: Depth map
            datatype.Pose: Pose (position, rotation)
        """
        for rgb, depth, pose in zip(self._rgb, self._depth, self._poses):
            yield rgb, depth, pose

    def get_rgb_stack_tensor(self) -> torch.Tensor:
        """Get the RGB images in the cache as a tensor of shape (N, C, H, W)

        Returns:
            torch.Tensor: RGB images stacked on the first dimension. in range [0, 1]
        """
        rgb_stack = np.stack(self._rgb, axis=-1).transpose(3, 2, 0, 1)
        return torch.Tensor(rgb_stack)
