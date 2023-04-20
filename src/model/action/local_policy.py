from abc import ABC, abstractmethod
from typing import List, Optional
from warnings import warn

import numpy as np
from habitat_sim.errors import GreedyFollowerError
from habitat_sim.agent import Agent
from habitat_sim.nav import GreedyGeodesicFollower, PathFinder
from yacs.config import CfgNode

from ...utils import datatypes


class LocalPolicy(ABC):
    @abstractmethod
    def __call__(self, global_goal: datatypes.Coordinate3D) -> List[Optional[datatypes.AgentAction]]:
        pass


class GreedyLocalPolicy(LocalPolicy):
    def __init__(self, local_policy_cfg: CfgNode, navmesh_filepath: str, agent: Agent):
        self._max_num_steps = local_policy_cfg.MAX_NUM_STEPS
        self._agent = agent
        self._path_finder = PathFinder()
        self._path_finder.load_nav_mesh(navmesh_filepath)
        self._planner = GreedyGeodesicFollower(self._path_finder, self._agent)

    def __call__(self, global_goal: datatypes.Coordinate3D) -> List[Optional[datatypes.AgentAction]]:
        global_goal_arr = np.array(global_goal)
        try:
            actions = self._planner.find_path(global_goal_arr)
        except GreedyFollowerError:
            warn("Greedy follower failed to find path. Returning no actions", RuntimeWarning)
            return [None]
        if len(actions) > self._max_num_steps + 1:
            actions = actions[: self._max_num_steps] + [None]
        return actions
