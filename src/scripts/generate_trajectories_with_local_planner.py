from fire import Fire
from typing import Tuple
from PIL import Image
from pathlib import Path
from src.data import scene as scene_module
from src import config
from src.model.action.local_policy import GreedyLocalPolicy
import numpy as np
import quaternion
import habitat_sim


def main(
    scene: str = "minival/00800-TEEsavR23oF",
    start_position: Tuple = (0.0, 0.0, 0.0),
    goal_position: Tuple = (-5.0, 0.0, -1.5),
):
    start_position = np.array(start_position)  # type: ignore[assignment]

    data_paths_cfg = config.default_data_paths_cfg()
    trajectories_dir = Path(data_paths_cfg.TRAJECTORIES_DIR)
    scene_split, scene_id = scene.split("/")
    scene_destination_dir = trajectories_dir / scene_split / scene_id
    navmesh_filepath = (
        Path(data_paths_cfg.RAW_DATA_DIR)
        / scene_split
        / "versioned_data/hm3d-0.2/hm3d"
        / scene_split
        / scene_id
        / f"{scene_id[6:]}.basis.navmesh"
    )

    rgb_dir = scene_destination_dir / "RGB"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    d_dir = scene_destination_dir / "D"
    d_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg = config.default_sim_cfg()
    sim = scene_module.initialize_sim(scene_split, scene_id, sim_cfg=sim_cfg)
    semantics_used = scene_module.check_if_semantic_sensor_used(sim)
    if semantics_used:
        sim_dir = scene_destination_dir / "Semantic"
        sim_dir.mkdir(parents=True, exist_ok=True)
    agent = sim.get_agent(sim_cfg.DEFAULT_AGENT_ID)
    agent_state = habitat_sim.AgentState()
    agent_state.position = start_position  # world space
    agent.set_state(agent_state)

    action_module_cfg = config.default_action_module_cfg()
    local_policy = GreedyLocalPolicy(action_module_cfg.LOCAL_POLICY, str(navmesh_filepath), agent)
    actions = local_policy(goal_position)  # type: ignore[arg-type]

    num_frames = len(actions)
    positions = np.empty((num_frames, 3), dtype=np.float64)
    rotations = np.empty((num_frames), dtype=np.quaternion)  # type: ignore[attr-defined]

    action = actions.pop(0)
    count = 0
    while action is not None:
        observations = sim.step(action)
        rgb = observations["color_sensor"]
        depth = observations["depth_sensor"]
        Image.fromarray(rgb[:, :, :3]).save(rgb_dir / f"{count}.png")
        np.save(d_dir / f"{count}", depth)
        if semantics_used:
            semantics = observations["semantic_sensor"]
            np.save(sim_dir / f"{count}", semantics)
        positions[count] = sim.get_agent(0).state.position
        rotations[count] = sim.get_agent(0).state.rotation
        count += 1
        action = actions.pop(0)
    
    np.save(scene_destination_dir / "positions", positions)
    np.save(scene_destination_dir / "rotations", rotations)

if __name__ == "__main__":
    Fire(main)
