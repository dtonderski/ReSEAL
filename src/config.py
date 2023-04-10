from pathlib import Path
from yacs.config import CfgNode


def default_map_builder_cfg() -> CfgNode:
    map_builder_cfg = CfgNode()
    map_builder_cfg.RESOLUTION = 1.0  # cm per pixel
    map_builder_cfg.EGOCENTRIC_MAP_SHAPE = (500, 500, 500)  # (x, y, z) in pixel
    map_builder_cfg.EGOCENTRIC_MAP_ORIGIN_OFFSET = (250, 250, 250)  # (x, y, z) in pixel
    map_builder_cfg.NUM_SEMANTIC_CLASSES = 10
    return map_builder_cfg

def default_data_paths_cfg() -> CfgNode:
    data_paths_cfg = CfgNode()
    data_paths_cfg.RAW_DATA_DIR = str(Path("data", "raw"))
    data_paths_cfg.ANNOTATED_SCENE_CONFIG_PATH_IN_SPLIT = str(Path("scene_datasets", "hm3d",
                                                                   "hm3d_annotated_basis.scene_dataset_config.json"))
    data_paths_cfg.BASIS_SCENE_DATASET_CONFIG_PATH_IN_SPLIT = str(Path("scene_datasets", "hm3d",
                                                                       "hm3d_basis.scene_dataset_config.json"))
    data_paths_cfg.INTERIM_DATA_DIR = str(Path("data", "interim"))
    data_paths_cfg.TRAJECTORIES_DIR = str(Path(data_paths_cfg.INTERIM_DATA_DIR, "trajectories"))

    return data_paths_cfg

def default_sim_cfg() -> CfgNode:
    sim_cfg = CfgNode()
    sim_cfg.SENSOR_CFG = default_sensor_cfg()
    sim_cfg.FORWARD_MOVE_DISPLACEMENT = 0.25 #m
    sim_cfg.TURN_ANGLE_DISPLACEMENT = 30 #deg
    sim_cfg.DEFAULT_AGENT_ID = 0
    sim_cfg.DEFAULT_POSITION = [-0.6, 0.0, 0.0] #m
    return sim_cfg

def default_sensor_cfg() -> CfgNode:
    sensor_cfg = CfgNode()
    sensor_cfg.WIDTH = 256 #px
    sensor_cfg.HEIGHT = 256 #px
    sensor_cfg.HFOV = 90 #deg
    sensor_cfg.SENSOR_HEIGHT = 0.88 #m
    return sensor_cfg
