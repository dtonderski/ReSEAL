from warnings import warn
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from habitat_sim.errors import GreedyFollowerError
from habitat_sim.agent import Agent
from habitat_sim.nav import GreedyGeodesicFollower, PathFinder
from yacs.config import CfgNode

from ...utils import datatypes


class LocalPolicy(ABC):
    @abstractmethod
    def __call__(self, global_goal: datatypes.Coordinate3D) -> Optional[datatypes.AgentAction]:
        pass


class GreedyLocalPolicy(LocalPolicy):
    def __init__(self, local_policy_cfg: CfgNode, navmesh_filepath: str, agent: Agent):
        self._agent = agent
        self._distance_threshold = local_policy_cfg.DISTANCE_THRESHOLD
        self._path_finder = PathFinder()
        self._path_finder.load_nav_mesh(navmesh_filepath)
        self._planner = GreedyGeodesicFollower(self._path_finder, self._agent)

    def __call__(self, global_goal: datatypes.Coordinate3D) -> Optional[datatypes.AgentAction]:
        if self._is_agent_within_threshold(global_goal):
            return None
        global_goal_arr = np.array(global_goal)
        try:
            action = self._planner.next_action_along(global_goal_arr)
        except GreedyFollowerError:
            warn("Greedy follower raised error. Ignoring and returning None.", RuntimeWarning)
            return None
        return action

    def _is_agent_within_threshold(self, global_goal: datatypes.Coordinate3D) -> bool:
        agent_position = self._agent.state.position
        global_goal_arr = np.array(global_goal)
        return np.linalg.norm(agent_position - global_goal_arr) < self._distance_threshold
