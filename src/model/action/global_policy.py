from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import habitat_sim
import torch
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from yacs.config import CfgNode

from .feature_extractor import SemanticMapFeatureExtractor


class RandomGlobalPolicy(ActorCriticPolicy):
    """Global policy that generates random goal coordinates based on given navmash
    Follows the same interface as ActorCriticPolicy of stable-baselines3
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
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

    def forward(
        self, _obs: torch.Tensor, _deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates random goal coordinates

        Returns:
            Tuple[Tensor, Tensor, Tensor]: goal coordinates, None, None
        """
        goal_position = self._path_finder.get_random_navigable_point()
        return torch.Tensor(goal_position), None, None  # type: ignore[return-value]

    def extract_features(self, obs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
        raise RuntimeError("RandomGlobalPolicy does not implement extract_features")

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        raise RuntimeError("RandomGlobalPolicy does not implement evaluate_actions")

    def get_distribution(self, obs: torch.Tensor) -> Distribution:
        raise RuntimeError("RandomGlobalPolicy does not implement get_distribution")

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("RandomGlobalPolicy does not implement predict_values")


def create_global_policy(
    global_policy_cfg: CfgNode,
    return_kwargs: bool = False,
    **kwargs,
) -> Union[Dict[str, Any], ActorCriticPolicy]:
    """Factory function for creating global policy."""
    lr_schedule = _create_lr_schedule(global_policy_cfg.LR_SCHEDULE)
    if global_policy_cfg.NAME == "RandomGlobalPolicy":
        if return_kwargs:
            raise RuntimeError("RandomGlobalPolicy does not support return_kwargs")
        return RandomGlobalPolicy(
            observation_space=kwargs["observation_space"],
            action_space=kwargs["action_space"],
            lr_schedule=lr_schedule,
            navmesh_filepath=kwargs["navmesh_filepath"],
        )
    if global_policy_cfg.NAME == "CnnPolicy":
        features_extractor_kwargs = dict(features_dim=256)
        if return_kwargs:
            return dict(
                features_extractor_class=SemanticMapFeatureExtractor,
                features_extractor_kwargs=features_extractor_kwargs,
            )
        return ActorCriticCnnPolicy(
            kwargs["observation_space"],
            kwargs["action_space"],
            lr_schedule,
            features_extractor_class=SemanticMapFeatureExtractor,
            features_extractor_kwargs=features_extractor_kwargs,
        )
    if global_policy_cfg.NAME == "LoadTrainedPolicy":
        if return_kwargs:
            raise RuntimeError("LoadTrainedPolicy does not support return_kwargs")
        return ActorCriticPolicy.load(global_policy_cfg.MODEL_PATH)
    raise RuntimeError(f"Unknown global policy: {global_policy_cfg.NAME}")


def _create_lr_schedule(lr_schedule_cfg: CfgNode) -> Schedule:
    if lr_schedule_cfg.NAME == "ConstantLR":
        return lambda _: lr_schedule_cfg.INIT_LR
    raise RuntimeError(f"Unknown learning rate schedule: {lr_schedule_cfg.NAME}")
