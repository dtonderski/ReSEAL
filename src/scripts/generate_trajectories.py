import json
from typing import Optional

import numpy as np
from fire import Fire
from habitat_sim.simulator import ObservationDict
from PIL import Image
from tqdm import trange

from src import config
from src.data import filepath, scene
from src.model.action import local_policy, pipeline
from src.utils import datatypes
from src.features.mapping import SemanticMap3DBuilder
from src.model.perception.model_wrapper import ModelWrapper


def main(
    scene_name: str = "minival/00800-TEEsavR23oF",
    start_position: datatypes.Coordinate3D = (0.0, 0.0, 0.0),
    max_num_steps: int = 200,
    goal_position: Optional[datatypes.Coordinate3D] = None,
    use_random_policy: bool = False,
    commands_file: Optional[str] = None,
    use_trained_policy: Optional[bool] = False,
):
    start_position = np.array(start_position)  # type: ignore[assignment]
    data_paths_cfg = config.default_data_paths_cfg()
    data_paths = filepath.GenerateTrajectoryFilepaths(data_paths_cfg, scene_name)
    # Create directories
    data_paths.rgb_dir.mkdir(parents=True, exist_ok=True)
    data_paths.depth_dir.mkdir(parents=True, exist_ok=True)

    # Initialize simulator
    sim_cfg = config.default_sim_cfg()
    sim_cfg.DEFAULT_POSITION = list(start_position)
    sim = scene.initialize_sim(
        data_paths.scene_split, data_paths.scene_id, data_paths_cfg=data_paths_cfg, sim_cfg=sim_cfg
    )
    agent = sim.get_agent(sim_cfg.DEFAULT_AGENT_ID)
    use_semantic_sensor = scene.check_if_semantic_sensor_used(sim)
    if use_semantic_sensor:
        data_paths.semantic_dir.mkdir(parents=True, exist_ok=True)

    # Initialize action pipeline
    action_module_cfg = config.default_action_module_cfg()
    if goal_position:
        greedy_policy = local_policy.GreedyLocalPolicy(
            action_module_cfg.LOCAL_POLICY, str(data_paths.navmesh_filepath), agent
        )
    elif use_random_policy:
        action_pipeline = pipeline.create_action_pipeline(action_module_cfg, str(data_paths.navmesh_filepath), agent)
    elif commands_file:
        with open(commands_file, "r", encoding="utf-8") as file:
            actions = json.load(file)
        if data_paths.scene_name not in actions:
            raise RuntimeError(f"Scene {data_paths.scene_name} not found in commands file.")
        actions = actions[data_paths.scene_name]
        max_num_steps = len(actions)
    elif use_trained_policy:
        # Initialize map builder and perception model
        map_builder_cfg = config.default_map_builder_cfg()
        perception_model_cfg = config.default_perception_model_cfg()
        map_builder = SemanticMap3DBuilder(map_builder_cfg, sim_cfg)
        perception_model = ModelWrapper(perception_model_cfg, device='cuda')
        # Initialize action pipeline
        action_module_cfg.PREPROCESSOR.NAME = "IdentityPreprocessor"
        action_module_cfg.GLOBAL_POLICY.NAME = "LoadTrainedPolicy"
        action_module_cfg.GLOBAL_POLICY.MAP_SHAPE = map_builder.semantic_map_at_pose_shape
        action_pipeline = pipeline.create_action_pipeline(action_module_cfg, str(data_paths.navmesh_filepath), agent)
    else:
        raise RuntimeError("No goal position, random policy, or commands file specified.")

    # Initialize output
    positions = np.empty((max_num_steps, 3), dtype=np.float64)
    rotations = np.empty((max_num_steps), dtype=np.quaternion)  # type: ignore[attr-defined]
    if use_random_policy:
        global_goals = np.empty((max_num_steps, 3), dtype=np.float64)

    # Run simulation
    for count in trange(max_num_steps):
        observations = sim.get_sensor_observations(0)
        rgb = observations["color_sensor"]  # pylint: disable=unsubscriptable-object
        depth = observations["depth_sensor"]  # pylint: disable=unsubscriptable-object
        Image.fromarray(rgb[:, :, :3]).save(data_paths.rgb_dir / f"{count}.png")
        np.save(data_paths.depth_dir / f"{count}", depth)
        if use_semantic_sensor:
            semantics = observations["semantic_sensor"]  # pylint: disable=unsubscriptable-object
            np.save(data_paths.semantic_dir / f"{count}", semantics)
        positions[count] = sim.get_agent(0).state.position
        rotations[count] = sim.get_agent(0).state.rotation
        if goal_position:
            action = greedy_policy(goal_position)
        elif use_random_policy:
            action = action_pipeline(None)  # type: ignore[arg-type]
            while not action:
                action = action_pipeline(None)  # type: ignore[arg-type]
            global_goals[count] = action_pipeline._global_goal  # pylint: disable=protected-access
        elif use_trained_policy:
            semantic_map_2d = perception_model(rgb[:, :, :3])  # type: ignore[arg-type,has-type]
            pose = (positions[count], rotations[count])
            map_builder.update_point_cloud(semantic_map_2d, depth, pose)  # type: ignore[arg-type,has-type]
            map_builder.update_semantic_map()
            semantic_map_3d = map_builder.semantic_map_at_pose(pose)
            action = action_pipeline(semantic_map_3d)
        else:
            action = actions[count]
        if action:
            _ = sim.step(action)

    # Save output
    np.save(data_paths.positions_filepath, positions)
    np.save(data_paths.rotations_filepath, rotations)
    if use_random_policy:
        np.save(data_paths.global_goals_filepath, global_goals)


if __name__ == "__main__":
    Fire(main)
