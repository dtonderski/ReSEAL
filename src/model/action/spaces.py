from gymnasium import spaces
import numpy as np
from yacs.config import CfgNode


def create_observation_space(global_policy_cfg: CfgNode) -> spaces.Space:
    """Factory function for creating observation space of the global policy. I.e. 3D semantic map
    
    Args:
        global_policy_cfg (CfgNode): global policy config

    Returns:
        spaces.Space: observation space
    """
    shape = global_policy_cfg.OBSERVATION_SPACE_SHAPE
    return spaces.Box(low=0, high=1, shape=shape)


def create_action_space() -> spaces.Space:
    """Factory function for creating action space of the global policy. I.e. goal coordinates

    Returns:
        spaces.Space: action space
    """
    return spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
