from typing import Optional, Tuple, Union

import habitat_sim
from gym import spaces
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from torch import Tensor
from yacs.config import CfgNode

from .spaces import create_action_space, create_observation_space


class RandomGlobalPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        navmesh_filepath: str,
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
        self._path_finder = habitat_sim.PathFinder()
        self._path_finder.load_nav_mesh(navmesh_filepath)

    def forward(self, _obs: Tensor, _deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        goal_position = self._path_finder.get_random_navigable_point()
        return Tensor(goal_position), None, None  # type: ignore[return-value]

    def extract_features(self, obs: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:  # type: ignore[override]
        raise RuntimeError("RandomGlobalPolicy does not implement extract_features")

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        raise RuntimeError("RandomGlobalPolicy does not implement evaluate_actions")

    def get_distribution(self, obs: Tensor) -> Distribution:
        raise RuntimeError("RandomGlobalPolicy does not implement get_distribution")

    def predict_values(self, obs: Tensor) -> Tensor:
        raise RuntimeError("RandomGlobalPolicy does not implement predict_values")


def create_global_policy(global_policy_cfg: CfgNode, **kwargs) -> ActorCriticPolicy:
    observation_space = create_observation_space(global_policy_cfg)
    action_space = create_action_space()
    lr_schedule = _create_lr_schedule(global_policy_cfg.LR_SCHEDULE)
    if global_policy_cfg.NAME == "RandomGlobalPolicy":
        return RandomGlobalPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            navmesh_filepath=kwargs["navmesh_filepath"],
        )
    raise RuntimeError(f"Unknown global policy: {global_policy_cfg.NAME}")


def _create_lr_schedule(lr_schedule_cfg: CfgNode) -> Schedule:
    if lr_schedule_cfg.NAME == "ConstantLR":
        return lambda _: lr_schedule_cfg.INIT_LR
    raise RuntimeError(f"Unknown learning rate schedule: {lr_schedule_cfg.NAME}")