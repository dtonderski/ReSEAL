from dataclasses import dataclass
from pathlib import Path

from yacs.config import CfgNode


@dataclass
class GenerateTrajectoryFilepaths:
    data_paths_cfg: CfgNode
    scene_name: str

    @property
    def scene_split(self) -> str:
        return self.scene_name.split("/", maxsplit=1)[0]

    @property
    def scene_id(self) -> str:
        return self.scene_name.split("/")[1]

    @property
    def trajectory_output_dir(self) -> Path:
        return Path(self.data_paths_cfg.TRAJECTORIES_DIR) / self.scene_name

    @property
    def rgb_dir(self) -> Path:
        return self.trajectory_output_dir / "RGB"

    @property
    def depth_dir(self) -> Path:
        return self.trajectory_output_dir / "D"

    @property
    def semantic_dir(self) -> Path:
        return self.trajectory_output_dir / "Semantic"

    @property
    def voxel_grid_dir(self) -> Path:
        return self.trajectory_output_dir / "Voxel_Grid"

    @property
    def top_down_semantic_map_dir(self) -> Path:
        return self.trajectory_output_dir / "Top_Down_Semantic_Map"

    @property
    def navmesh_filepath(self) -> Path:
        return (
            Path(self.data_paths_cfg.RAW_DATA_DIR)
            / self.scene_split
            / "versioned_data/hm3d-0.2/hm3d"
            / self.scene_split
            / self.scene_id
            / f"{self.scene_id[6:]}.basis.navmesh"
        )

    @property
    def positions_filepath(self) -> Path:
        return self.trajectory_output_dir / "positions.npy"

    @property
    def rotations_filepath(self) -> Path:
        return self.trajectory_output_dir / "rotations.npy"

    @property
    def global_goals_filepath(self) -> Path:
        return self.trajectory_output_dir / "global_goals.npy"
