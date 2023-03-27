import json
import os
import pathlib
from PIL import Image
import numpy as np
import quaternion # type: ignore # pylint: disable=unused-import
from habitat_sim.simulator import ObservationDict
from src.data import scene

def main() -> None:
    with open('config/trajectories.json', 'r', encoding="utf-8") as file:
        json_dict = json.load(file)

    base_destination_dir = 'data/interim/trajectories/train'

    for scene_path, actions in json_dict.items():
        scene_destination_dir = os.path.join(base_destination_dir, pathlib.PurePath(scene_path).parent.name)

        os.makedirs(os.path.join(scene_destination_dir, "RGB"),
                    exist_ok=True)
        os.makedirs(os.path.join(scene_destination_dir, "D"),
                    exist_ok=True)

        sim = scene.initialize_scene(f"data/raw/train/scene_datasets/hm3d/train/{scene_path}")
        print(f"Generating trajectory from scene {scene_path} and saving to {scene_destination_dir}")

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
                Image.fromarray(rgb[:,:,:3]).save(os.path.join(scene_destination_dir, "RGB", f"{current_frame}.png"))

                # Save depth
                np.save(os.path.join(scene_destination_dir, "D", f"{current_frame}"), depth)

                positions[current_frame] = sim.get_agent(0).state.position
                rotations[current_frame] = sim.get_agent(0).state.rotation

                observations = sim.step(actions[current_frame])
                current_frame = current_frame + 1
            except KeyboardInterrupt:
                break

        # Save positions and rotations
        np.save(os.path.join(scene_destination_dir, "positions"), positions)
        np.save(os.path.join(scene_destination_dir, "rotations"), rotations)
if __name__ == '__main__':
    main()
