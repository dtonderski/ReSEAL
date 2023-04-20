from yacs.config import CfgNode
from stable_baselines3.common.policies import ActorCriticPolicy

from ...utils import datatypes
from .local_policy import LocalPolicy
from .preprocessing import create_preprocessor, SemanticMapPreprocessor
from .global_policy import create_global_policy


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
        local_policy: LocalPolicy,
        action_pipeline_cfg: CfgNode,
    ) -> None:
        self._semantic_map_preprocessor = semantic_map_preprocessor
        self._global_policy = global_policy
        self._local_policy = local_policy
        self._is_deterministic = action_pipeline_cfg.INFERENCE.IS_DETERMINISTIC
        self._global_goal: datatypes.Coordinate3D = (0, 0, 0)
        self._counter = _Counter(action_pipeline_cfg.GLOBAL_POLICY_POLLING_FREQUENCY)

    def __call__(self, semantic_map: datatypes.SemanticMap3D) -> datatypes.AgentAction:
        processed_map = self._semantic_map_preprocessor(semantic_map)
        if self._counter.is_zero():
            global_goal, _, _ = self._global_policy.forward(processed_map, True)
            self._global_goal = tuple(global_goal.numpy(force=True))  # type: ignore[assignment]
        self._counter.step()
        return self._local_policy(self._global_goal)


def create_action_pipeline(action_module_cfg: CfgNode) -> ActionPipeline:
    semantic_map_preprocessor = create_preprocessor(action_module_cfg.PREPROCESSOR)
    global_policy = create_global_policy(action_module_cfg.GLOBAL_POLICY)
    # TODO: Use local policy after it is implemented
    local_policy = LocalPolicy() # type: ignore[abstract] 
    return ActionPipeline(semantic_map_preprocessor, global_policy, local_policy, action_module_cfg.ACTION_PIPELINE)
