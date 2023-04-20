from typing import Optional

from habitat_sim.agent import Agent
from stable_baselines3.common.policies import ActorCriticPolicy
from yacs.config import CfgNode

from ...utils import datatypes
from .global_policy import create_global_policy
from .local_policy import GreedyLocalPolicy, LocalPolicy
from .preprocessing import SemanticMapPreprocessor, create_preprocessor


class _Counter:
    def __init__(self, max: int):
        self._max = max
        self._counter = 0

    def step(self) -> None:
        self._counter = (self._counter + 1) % self._max

    def is_zero(self) -> bool:
        return self._counter == 0


class ActionPipeline:
    def __init__(
        self,
        semantic_map_preprocessor: SemanticMapPreprocessor,
        global_policy: ActorCriticPolicy,
        local_policy: local_policy_module.LocalPolicy,
        action_pipeline_cfg: CfgNode,
    ) -> None:
        self._semantic_map_preprocessor = semantic_map_preprocessor
        self._global_policy = global_policy
        self._local_policy = local_policy
        self._is_deterministic = action_pipeline_cfg.IS_DETERMINISTIC
        self._global_goal: datatypes.Coordinate3D = (0, 0, 0)
        self._actions_queue: List[Optional[datatypes.AgentAction]] = [None]

    def __call__(self, semantic_map: datatypes.SemanticMap3D) -> Optional[datatypes.AgentAction]:
        preprocessed_semantic_map = self._semantic_map_preprocessor(semantic_map)
        if self._counter.is_zero():
            global_goal, _, _ = self._global_policy(preprocessed_semantic_map, self._is_deterministic)
            self._global_goal = tuple(global_goal.numpy(force=True))  # type: ignore[assignment]
            self._actions_queue = self._local_policy(self._global_goal)
            action = self._actions_queue.pop()
        if action is None:
            raise RuntimeError("Local policy returned no action")
        return action


def create_action_pipeline(action_module_cfg: CfgNode, navmesh_filepath: str, agent: Agent) -> ActionPipeline:
    semantic_map_preprocessor = create_preprocessor(action_module_cfg.PREPROCESSOR)
    global_policy = create_global_policy(action_module_cfg.GLOBAL_POLICY, navmesh_filepath=navmesh_filepath)
    local_policy = GreedyLocalPolicy(action_module_cfg.LOCAL_POLICY, navmesh_filepath, agent)
    return ActionPipeline(semantic_map_preprocessor, global_policy, local_policy, action_module_cfg.ACTION_PIPELINE)
