from gymnasium import spaces
import numpy as np
from yacs.config import CfgNode


def create_observation_space(global_policy_cfg: CfgNode) -> spaces.Space:
    shape = global_policy_cfg.OBSERVATION_SPACE_SHAPE
    return spaces.Box(low=0, high=1, shape=shape)


def create_action_space() -> spaces.Space:
    """
    Action space of the global policy.
    - position: 3D position (i.e. spaces.Box with shape (3,), bounds=[-inf, inf])
    """
    return spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
