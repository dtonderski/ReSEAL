from typing import List, Tuple, Union

import gymnasium as gym
import habitat_sim
import numpy as np


def create_observation_space(map_shape: Tuple[int, int, int, int]) -> gym.spaces.Dict:
    """Factory function for creating observation space of the global policy. I.e. 3D semantic map

    Args:
        map_shape (Tuple[int, int, int, int]): Shape of the semantic map (height, width, depth, channels)

    Returns:
        spaces.Space: observation space
    """
    return gym.spaces.Dict(
        {
            "map": gym.spaces.Box(0, 1, shape=(map_shape[3], map_shape[0], map_shape[1], map_shape[2])),
            "position": gym.spaces.Box(-50, 50, shape=(3,)),
        }
    )


def create_action_space(navmesh_filepath: Union[str, List[str]]) -> gym.spaces.Space:
    """Factory function for creating action space of the global policy. I.e. goal coordinates

    Returns:
        spaces.Space: action space
    """
    if isinstance(navmesh_filepath, str):
        navmesh_filepath = [navmesh_filepath]
    bounds = []
    for navmesh in navmesh_filepath:
        path_finder = habitat_sim.PathFinder()
        path_finder.load_nav_mesh(navmesh)
        min_point, max_point = path_finder.get_bounds()
        bounds.extend([min_point, max_point])
    bounds = np.concatenate(bounds)
    min_coord = np.min(bounds)
    max_coord = np.max(bounds)
    return gym.spaces.Box(min_coord, max_coord, shape=(3,))
