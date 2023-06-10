from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import habitat_sim
import torch
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common import policies
from stable_baselines3.common.type_aliases import Schedule
from yacs.config import CfgNode

from .feature_extractor import SemanticMapFeatureExtractor
from .spaces import create_action_space, create_observation_space


class RandomGlobalPolicy(policies.ActorCriticPolicy):
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
        action = torch.Tensor(goal_position)
        action = action.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return action, None, None  # type: ignore[return-value]

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
    navmesh_filepath: str,
    return_kwargs: bool = False,
    **kwargs,
) -> Union[Dict[str, Any], policies.ActorCriticPolicy]:
    """Factory function for creating global policy."""
    observation_space = create_observation_space(global_policy_cfg.MAP_SHAPE)
    action_space = create_action_space(navmesh_filepath)
    lr_schedule = _create_lr_schedule(global_policy_cfg.LR_SCHEDULE)

    if global_policy_cfg.NAME == "RandomGlobalPolicy":
        if return_kwargs:
            raise RuntimeError("RandomGlobalPolicy does not support return_kwargs")
        return RandomGlobalPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            navmesh_filepath=navmesh_filepath,
        )

    if global_policy_cfg.NAME == "MultiInputPolicy":
        features_extractor_kwargs = dict(features_dim=512)
        if return_kwargs:
            return dict(
                features_extractor_class=SemanticMapFeatureExtractor,
                features_extractor_kwargs=features_extractor_kwargs,
            )
        device = torch.device("cuda")
        return policies.MultiInputActorCriticPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=SemanticMapFeatureExtractor,
            features_extractor_kwargs=features_extractor_kwargs,
        ).to(device)

    if global_policy_cfg.NAME == "LoadTrainedPolicy":
        if return_kwargs:
            raise RuntimeError("LoadTrainedPolicy does not support return_kwargs")
        return policies.MultiInputActorCriticPolicy.load(global_policy_cfg.MODEL_PATH)

    raise RuntimeError(f"Unknown global policy: {global_policy_cfg.NAME}")


def _create_lr_schedule(lr_schedule_cfg: CfgNode) -> Schedule:
    if lr_schedule_cfg.NAME == "ConstantLR":
        return lambda _: lr_schedule_cfg.INIT_LR
    raise RuntimeError(f"Unknown learning rate schedule: {lr_schedule_cfg.NAME}")
