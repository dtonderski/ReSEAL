from typing import Tuple

import gymnasium as gym


def create_observation_space(map_shape: Tuple[int, int, int, int]) -> gym.spaces.Dict:
    """Factory function for creating observation space of the global policy. I.e. 3D semantic map

    Args:
        map_shape (Tuple[int, int, int, int]): Shape of the semantic map (height, width, depth, channels)

    Returns:
        spaces.Space: observation space
    """
    return gym.spaces.Dict({
        "map": gym.spaces.Box(0, 1, shape=(map_shape[3], map_shape[0], map_shape[1], map_shape[2])),
        "position": gym.spaces.Box(-50, 50, shape=(3,)),
    })


def create_action_space() -> gym.spaces.Space:
    """Factory function for creating action space of the global policy. I.e. goal coordinates

    Returns:
        spaces.Space: action space
    """
    return gym.spaces.Box(
        -50, 50, shape=(3,)
    )  # TODO: The bounds are arbitrary large numbers. Technically this could be determined from the scene.
