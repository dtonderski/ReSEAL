from fire import Fire

import numpy as np
import cv2
from tqdm import trange
import quaternion  # pylint: disable=unused-import

from src.data import filepath
from src.features import mapping
from src import config


def main(
    scene_name: str = "minival/00800-TEEsavR23oF",
    num_steps: int = 100,
):
    datapaths_cfg = config.default_data_paths_cfg()
    data_paths = filepath.GenerateTrajectoryFilepaths(datapaths_cfg, scene_name)
    data_paths.voxel_grid_dir.mkdir(parents=True, exist_ok=True)
    data_paths.top_down_semantic_map_dir.mkdir(parents=True, exist_ok=True)

    # Initialize map builder
    map_builder_cfg = config.default_map_builder_cfg()
    map_builder_cfg.NUM_SEMANTIC_CLASSES = 3
    map_builder_cfg.RESOLUTION = 0.05
    map_builder_cfg.MAP_SIZE = [2.0, 2.0, 2.0]
    sim_cfg = config.default_sim_cfg()
    map_builder = mapping.SemanticMap3DBuilder(map_builder_cfg, sim_cfg)

    # Load positions and rotations
    positions = np.load(data_paths.positions_filepath)
    rotations = np.load(data_paths.rotations_filepath).view(np.quaternion)  # type: ignore[attr-defined]
    num_positions = positions.shape[0]
    num_steps = min(num_positions, num_steps)

    # Build point cloud
    if data_paths.semantic_dir.exists():
        colors = np.random.random((1000, 3))
    for i in trange(num_steps):
        depth_map = np.load(data_paths.depth_dir / f"{i}.npy")
        if data_paths.semantic_dir.exists():
            semantic_map = np.load(data_paths.semantic_dir / f"{i}.npy")
            semantic_map = colors[semantic_map, :]
        else:
            # Use RGB as placeholder for semantic map
            semantic_map = cv2.imread(str(data_paths.rgb_dir / f"{i}.png"))  # pylint: disable=no-member
            semantic_map = cv2.cvtColor(semantic_map, cv2.COLOR_BGR2RGB) / 255  # pylint: disable=no-member
        pose = (positions[i], rotations[i])
        map_builder.update_point_cloud(semantic_map, depth_map, pose)
    map_builder.update_semantic_map()

    # Save map
    for i in trange(num_steps):
        pose = (positions[i], rotations[i])
        semantic_voxel_grid = map_builder.semantic_map_at_pose(pose)
        np.save(data_paths.voxel_grid_dir / f"{i}.npy", semantic_voxel_grid)
        # Save image of map from top down
        top_down_map = semantic_voxel_grid.max(axis=1)[:,:,1:] * 255
        top_down_map = top_down_map.astype(np.uint8)
        top_down_map = cv2.cvtColor(top_down_map, cv2.COLOR_RGB2BGR)  # pylint: disable=no-member
        top_down_map = cv2.resize(top_down_map, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)  # pylint: disable=no-member
        cv2.imwrite(str(data_paths.top_down_semantic_map_dir / f"{i}.png"), top_down_map) # pylint: disable=no-member


if __name__ == "__main__":
    Fire(main)
