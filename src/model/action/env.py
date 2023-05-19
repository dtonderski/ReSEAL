from typing import Dict, Optional, Tuple

import gymnasium as gym
import habitat_sim
import numpy as np
from yacs.config import CfgNode

from ...features.mapping import SemanticMap3DBuilder
from ...utils import datatypes
from ..perception.model_wrapper import ModelWrapper
from .local_policy import LocalPolicy
from .preprocessing import SemanticMapPreprocessor
from .spaces import create_action_space, create_observation_space


class HabitatEnv(gym.Env):
    """This class wraps the habitat-sim simulator, perception pipeline and map builder,
    and provides an OpenAI Gym interface.

    Args:
        sim (habitat_sim.Simulator): The habitat-sim simulator
        local_policy (LocalPolicy): The local policy (e.g. GreedyLocalPolicy)
        map_builder (SemanticMap3DBuilder): The map builder
        model (ModelWrapper): The perception model
        cfg (CfgNode): Environment configuration
        navmesh_filepath (str, optional): Path to the navmesh. Defaults to None.
            If not None, the agent is initialized to a random point when reset is called
    """

    def __init__(
        self,
        sim: habitat_sim.Simulator,
        local_policy: LocalPolicy,
        map_builder: SemanticMap3DBuilder,
        perception_model: ModelWrapper,
        preprocessor: SemanticMapPreprocessor,
        cfg: CfgNode,
        navmesh_filepath: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._sim = sim
        self._map_builder = map_builder
        self._local_policy = local_policy
        self._perception_model = perception_model
        self._preprocessor = preprocessor
        self._path_finder = None
        if navmesh_filepath:
            self._path_finder = habitat_sim.PathFinder()
            self._path_finder.load_nav_mesh(navmesh_filepath)
        self._cfg = cfg
        self.observation_space = create_observation_space(self._map_builder.semantic_map_at_pose_shape)
        self.action_space = create_action_space()
        self._counter = 0

    def step(self, action) -> Tuple[datatypes.SemanticMap3D, float, bool, bool, Dict]:
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

    def reset(self, seed=None, options=None) -> Tuple[datatypes.SemanticMap3D, Dict]:
        """Reset the environment. This resets the simulator, map builder, and counter.
        # TODO: Initialize the agent at a random pose

        Returns:
            datatypes.SemanticMap3D: The semantic map at the agent's initial pose
            dict: Additional info
        """
        # Reset
        self._sim.reset()
        self._map_builder.clear()
        self._counter = 0
        # Set agent to random pose
        if self._path_finder:
            new_position = self._path_finder.get_random_navigable_point()
            self._sim.get_agent(0).set_state(habitat_sim.agent.AgentState(new_position))
        # Get observations
        self._update_obs()
        obs = self._get_obs()
        info = {}  # type: ignore[var-annotated]
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
        semantic_map = self._perception_model(rgb[:, :, :3])
        pose = (position, rotation)
        self._map_builder.update_point_cloud(semantic_map, depth_map, pose)  # type: ignore[arg-type]

    def _get_obs(self) -> datatypes.SemanticMap3D:
        """Gets the semantic map at the current agent pose from the map builder"""
        self._map_builder.update_semantic_map()
        position = self._sim.get_agent(0).state.position
        rotation = self._sim.get_agent(0).state.rotation
        pose = (position, rotation)
        semantic_map = self._map_builder.semantic_map_at_pose(pose)
        return self._preprocessor(semantic_map).numpy(force=True)

    def _gainful_curiosity(self) -> float:
        """Reward function, defined as the number of voxels in the semantic map with semantic labels
        whose confidence is above a threshold"""
        semantic_map = self._map_builder.semantic_map
        return np.sum(semantic_map[:, :, :, 1:] >= self._cfg.GAINFUL_CURIOUSITY_THRESHOLD)
