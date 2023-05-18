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


def main(
    scene_name: str = "minival/00800-TEEsavR23oF",
    start_position: datatypes.Coordinate3D = (0.0, 0.0, 0.0),
    max_num_steps: int = 200,
    goal_position: Optional[datatypes.Coordinate3D] = None,
    use_random_policy: bool = False,
    commands_file: Optional[str] = None,
    epoch_number: Optional[int] = None
):
    start_position = np.array(start_position)  # type: ignore[assignment]
    data_paths_cfg = config.default_data_paths_cfg()
    if epoch_number is not None:
        data_paths_cfg.TRAJECTORIES_DIR = data_paths_cfg.TRAJECTORIES_DIR + "/epoch_"+str(epoch_number) 
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
    if goal_position:
        action_module_cfg = config.default_action_module_cfg()
        greedy_policy = local_policy.GreedyLocalPolicy(
            action_module_cfg.LOCAL_POLICY, str(data_paths.navmesh_filepath), agent
        )
    elif use_random_policy:
        action_module_cfg = config.default_action_module_cfg()
        action_pipeline = pipeline.create_action_pipeline(action_module_cfg, str(data_paths.navmesh_filepath), agent)
    elif commands_file:
        with open(commands_file, "r", encoding="utf-8") as file:
            actions = json.load(file)
        if data_paths.scene_name not in actions:
            raise RuntimeError(f"Scene {data_paths.scene_name} not found in commands file.")
        actions = actions[data_paths.scene_name]
        max_num_steps = len(actions)
    else:
        raise RuntimeError("No goal position, random policy, or commands file specified.")

    # Initialize output
    positions = np.empty((max_num_steps, 3), dtype=np.float64)
    rotations = np.empty((max_num_steps), dtype=np.quaternion)  # type: ignore[attr-defined]
    if use_random_policy:
        global_goals = np.empty((max_num_steps, 3), dtype=np.float64)

    # Run simulation
    count = 0
    for count in trange(max_num_steps):
        if goal_position:
            action = greedy_policy(goal_position)
        elif use_random_policy:
            action = action_pipeline(None)  # type: ignore[arg-type]
            while not action:
                action = action_pipeline(None)  # type: ignore[arg-type]
            global_goals[count] = action_pipeline._global_goal  # pylint: disable=protected-access
        else:
            action = actions[count]
        if not action:
            break
        observations: ObservationDict = sim.step(action)
        rgb = observations["color_sensor"]  # pylint: disable=unsubscriptable-object
        depth = observations["depth_sensor"]  # pylint: disable=unsubscriptable-object
        Image.fromarray(rgb[:, :, :3]).save(data_paths.rgb_dir / f"{count}.png")
        np.save(data_paths.depth_dir / f"{count}", depth)
        if use_semantic_sensor:
            semantics = observations["semantic_sensor"]  # pylint: disable=unsubscriptable-object
            np.save(data_paths.semantic_dir / f"{count}", semantics)
        positions[count] = sim.get_agent(0).state.position
        rotations[count] = sim.get_agent(0).state.rotation

    # Save output
    np.save(data_paths.positions_filepath, positions)
    np.save(data_paths.rotations_filepath, rotations)
    if use_random_policy:
        np.save(data_paths.global_goals_filepath, global_goals)


if __name__ == "__main__":
    Fire(main)
