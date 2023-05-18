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
    """Encapsulates the entire action pipeline, including semantic map preprocessing, global policy and local policy.
    Use the factory function create_action_pipeline to create the action pipeline
    
    Args:
        semantic_map_preprocessor (SemanticMapPreprocessor): Semantic map preprocessor
        global_policy (ActorCriticPolicy): Global policy
        local_policy (LocalPolicy): Local policy
        action_pipeline_cfg (CfgNode): Action pipeline configuration
    """
    def __init__(
        self,
        semantic_map_preprocessor: SemanticMapPreprocessor,
        global_policy: ActorCriticPolicy,
        local_policy: LocalPolicy,
        action_pipeline_cfg: CfgNode,
    ) -> None:
        self._semantic_map_preprocessor = semantic_map_preprocessor
        self._global_policy = global_policy
        self._local_policy = local_policy
        self._is_deterministic = action_pipeline_cfg.IS_DETERMINISTIC
        self._global_goal: datatypes.Coordinate3D = (0, 0, 0)
        self._counter = _Counter(action_pipeline_cfg.GLOBAL_POLICY_POLLING_FREQUENCY)

    def __call__(self, semantic_map: datatypes.SemanticMap3D) -> Optional[datatypes.AgentAction]:
        """Given a semantic map, generates agent action to reach the goal. If not possible, returns None
        
        Args:
            semantic_map (datatypes.SemanticMap3D): Semantic map

        Returns:
            Optional[datatypes.AgentAction]: Agent action to reach the goal
        """
        preprocessed_semantic_map = self._semantic_map_preprocessor(semantic_map)
        if self._counter.is_zero():
            global_goal, _, _ = self._global_policy(preprocessed_semantic_map, self._is_deterministic)
            self._global_goal = tuple(global_goal.numpy(force=True))  # type: ignore[assignment]
        self._counter.step()
        return self._local_policy(self._global_goal)


def create_action_pipeline(action_module_cfg: CfgNode, navmesh_filepath: str, agent: Agent) -> ActionPipeline:
    """Factory function to create the action pipeline
    
    Args:
        action_module_cfg (CfgNode): Action module configuration
        navmesh_filepath (str): Path to the navmesh file
        agent (Agent): Habitat agent
    
    Returns:
        ActionPipeline: Action pipeline
    """
    semantic_map_preprocessor = create_preprocessor(action_module_cfg.PREPROCESSOR)
    global_policy = create_global_policy(action_module_cfg.GLOBAL_POLICY, navmesh_filepath=navmesh_filepath)
    local_policy = GreedyLocalPolicy(action_module_cfg.LOCAL_POLICY, navmesh_filepath, agent)
    return ActionPipeline(semantic_map_preprocessor, global_policy, local_policy, action_module_cfg.ACTION_PIPELINE)
