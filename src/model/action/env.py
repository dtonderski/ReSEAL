from typing import Tuple

import gymnasium as gym
import habitat_sim
import numpy as np
from yacs.config import CfgNode

from ...data.scene import initialize_sim
from ...features.mapping import SemanticMap3DBuilder
from ...utils import datatypes
from ..perception.model_wrapper import ModelWrapper
from .local_policy import LocalPolicy


class HabitatEnv(gym.Env):
    """This class wraps the habitat-sim simulator, perception pipeline and map builder,
    and provides an OpenAI Gym interface.
    
    Args:
        sim (habitat_sim.Simulator): The habitat-sim simulator
        local_policy (LocalPolicy): The local policy (e.g. GreedyLocalPolicy)
        map_builder (SemanticMap3DBuilder): The map builder
        model (ModelWrapper): The perception model
        cfg (CfgNode): Environment configuration
    """
    def __init__(
        self,
        sim: habitat_sim.Simulator,
        local_policy: LocalPolicy,
        map_builder: SemanticMap3DBuilder,
        model: ModelWrapper,
        cfg: CfgNode,
    ) -> None:
        super().__init__()
        self._sim = sim
        self._map_builder = map_builder
        self._local_policy = local_policy
        self._model = model
        self._cfg = cfg
        map_shape = self._map_builder.semantic_map_at_pose_size
        self.observation_space = gym.spaces.Box(0, 1, shape=(map_shape[3], map_shape[0], map_shape[1], map_shape[2]))
        self.action_space = gym.spaces.Box(-1000, 1000, shape=(3,)) # TODO: The bounds are arbitrary large numbers. Technically this could be determined from the scene.
        self._counter = 0

    def step(self, action) -> Tuple[datatypes.SemanticMap3D, float, bool, bool, dict]:
        """Give global_policy action (ie goal coordinates), agent tries to navigate to the goal with the local policy
        for a fixed number of steps (GLOBAL_POLICY_POLLING_FREQUENCY).
        After each step, the agent's observations are updated (ie the map builder's point cloud is updated.
        Then the resulting semantic map (VoxelGrid) is calculated at the agent's final pose. This is the observation
        The reward is calculated as the gainful curiosity.
        If the number of global steps exceeds MAX_STEPS, the episode is done.

        Returns:
            datatypes.SemanticMap3D: The semantic map at the agent's final pose
            float: The reward
            bool: Whether the episode is done
            bool: truncated (always False)
            dict: Additional info
        """
        for _ in range(self._cfg.GLOBAL_POLICY_POLLING_FREQUENCY):
            agent_action = self._local_policy(action)
            if agent_action:
                _ = self._sim.step(agent_action)
            self._update_obs()
        self._counter += 1
        obs = self._get_obs()
        reward = self._gainful_curiosity()
        done = self._counter >= self._cfg.MAX_STEPS
        # TODO: Do we need to add info to the env?
        info = {}  # type: ignore[var-annotated]
        return obs, reward, done, False, info

    def reset(self, _seed=None, _options=None):
        """Reset the environment. This resets the simulator, map builder, and counter.
        # TODO: Initialize the agent at a random pose
        
        Returns:
            datatypes.SemanticMap3D: The semantic map at the agent's initial pose
            dict: Additional info
        """
        self._sim.reset()
        self._map_builder.clear()
        self._counter = 0
        self._update_obs()
        obs = self._get_obs()
        info = {}
        return obs, info

    def render(self):
        # TODO: Render: save images?
        pass

    def close(self) -> None:
        self._sim.close()

    def _update_obs(self) -> None:
        """Gets current agent observations from the simulaotr,
        and updates the map builder (just the point cloud)"""
        observations = self._sim.get_sensor_observations(0)
        rgb = observations["color_sensor"]  # pylint: disable=unsubscriptable-object
        depth_map = observations["depth_sensor"]  # pylint: disable=unsubscriptable-object
        position = self._sim.get_agent(0).state.position
        rotation = self._sim.get_agent(0).state.rotation
        semantic_map = self._model(rgb[:,:,:3])
        pose = (position, rotation)
        self._map_builder.update_point_cloud(semantic_map, depth_map, pose)  # type: ignore[arg-type]

    def _get_obs(self) -> datatypes.SemanticMap3D:
        """Gets the semantic map at the current agent pose from the map builder"""
        self._map_builder.update_semantic_map()
        position = self._sim.get_agent(0).state.position
        rotation = self._sim.get_agent(0).state.rotation
        pose = (position, rotation)
        semantic_map = self._map_builder.semantic_map_at_pose(pose)
        semantic_map = np.transpose(semantic_map, (3, 0, 1, 2))
        return semantic_map

    def _gainful_curiosity(self) -> float:
        """Reward function, defined as the number of voxels in the semantic map with semantic labels
        whose confidence is above a threshold"""
        semantic_map = self._map_builder.semantic_map
        return np.sum(semantic_map[:, :, :, 1:] >= self._cfg.GAINFUL_CURIOUSITY_THRESHOLD)
