import json
from pathlib import Path
from PIL import Image

import numpy as np
import quaternion  # type: ignore # pylint: disable=unused-import
from habitat_sim.simulator import ObservationDict
from PIL import Image

from src.data import scene
from src.config import default_data_paths_cfg


def main() -> None:

    with open('config/trajectories.json', 'r', encoding="utf-8") as file:
        json_dict = json.load(file)

    data_paths_cfg = default_data_paths_cfg()

    trajectories_dir = Path(data_paths_cfg.TRAJECTORIES_DIR)

    for scene_info, actions in json_dict.items():

        scene_split,scene_id = scene_info.split('/')
        scene_destination_dir = trajectories_dir / scene_split / scene_id

        rgb_dir = scene_destination_dir / "RGB"
        rgb_dir.mkdir(parents=True, exist_ok=True)

        d_dir = scene_destination_dir / "D"
        d_dir.mkdir(parents=True, exist_ok=True)

        sim = scene.initialize_sim(scene_split, scene_id)

        semantics_used = scene.check_if_semantic_sensor_used(sim)
        if semantics_used:
            sim_dir = scene_destination_dir / "Semantic"
            sim_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating trajectory from scene {scene_info} and saving to {scene_destination_dir}")

        current_frame = 0

        max_frames = len(actions)
        observations: ObservationDict = sim.step('move_forward')

        positions = np.empty((max_frames, 3), dtype=np.float64)
        rotations = np.empty((max_frames), dtype=np.quaternion) # type: ignore[attr-defined]

        while current_frame < max_frames:
            try:
                rgb = observations["color_sensor"] # pylint: disable=unsubscriptable-object
                depth = observations["depth_sensor"] # pylint: disable=unsubscriptable-object

                # Only get RGB, discard last dimension
                Image.fromarray(rgb[:,:,:3]).save(rgb_dir / f"{current_frame}.png")

                # Save depth
                np.save(d_dir / f"{current_frame}", depth)

                # Save semantic
                if semantics_used:
                    semantics = observations["semantic_sensor"] # pylint: disable=unsubscriptable-object
                    np.save(sim_dir / f"{current_frame}", semantics)

                positions[current_frame] = sim.get_agent(0).state.position
                rotations[current_frame] = sim.get_agent(0).state.rotation

                observations = sim.step(actions[current_frame])
                current_frame = current_frame + 1
            except KeyboardInterrupt:
                break

        # Save positions and rotations
        np.save(scene_destination_dir / "positions", positions)
        np.save(scene_destination_dir / "rotations", rotations)
        

if __name__ == '__main__':
    main()
