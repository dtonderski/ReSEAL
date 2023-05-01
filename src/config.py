from pathlib import Path

from yacs.config import CfgNode


def default_map_builder_cfg() -> CfgNode:
    map_builder_cfg = CfgNode()
    map_builder_cfg.RESOLUTION = 0.05  # m per pixel
    map_builder_cfg.MAP_SIZE = (5.0, 2.0, 5.0)  # (x, y, z) in m
    map_builder_cfg.NUM_SEMANTIC_CLASSES = 10
    return map_builder_cfg


def default_data_paths_cfg() -> CfgNode:
    data_paths_cfg = CfgNode()
    data_paths_cfg.RAW_DATA_DIR = str(Path("data", "raw"))
    data_paths_cfg.ANNOTATED_SCENE_CONFIG_PATH_IN_SPLIT = str(
        Path("scene_datasets", "hm3d", "hm3d_annotated_basis.scene_dataset_config.json")
    )
    data_paths_cfg.BASIS_SCENE_DATASET_CONFIG_PATH_IN_SPLIT = str(
        Path("scene_datasets", "hm3d", "hm3d_basis.scene_dataset_config.json")
    )
    data_paths_cfg.INTERIM_DATA_DIR = str(Path("data", "interim"))
    data_paths_cfg.TRAJECTORIES_DIR = str(Path(data_paths_cfg.INTERIM_DATA_DIR, "trajectories"))

    return data_paths_cfg


def default_sim_cfg() -> CfgNode:
    sim_cfg = CfgNode()
    sim_cfg.SENSOR_CFG = default_sensor_cfg()
    sim_cfg.FORWARD_MOVE_DISPLACEMENT = 0.25 #m
    sim_cfg.TURN_ANGLE_DISPLACEMENT = 30 #deg
    sim_cfg.DEFAULT_AGENT_ID = 0
    sim_cfg.DEFAULT_POSITION = [-0.6, 0.0, 0.0]  # m
    sim_cfg.SENSOR_CFG = CfgNode()
    sim_cfg.SENSOR_CFG.WIDTH = 256  # px
    sim_cfg.SENSOR_CFG.HEIGHT = 256  # px
    sim_cfg.SENSOR_CFG.HFOV = 90  # px

    return sim_cfg

def default_sensor_cfg() -> CfgNode:
    sensor_cfg = CfgNode()
    sensor_cfg.WIDTH = 256 #px
    sensor_cfg.HEIGHT = 256 #px
    sensor_cfg.HFOV = 90 #deg
    sensor_cfg.SENSOR_HEIGHT = 0.88 #m
    return sensor_cfg

def default_action_module_cfg() -> CfgNode:
    action_module_cfg = CfgNode()
    # Config for semantic map preprocessor
    action_module_cfg.PREPROCESSOR = CfgNode()
    action_module_cfg.PREPROCESSOR.NAME = "DummyPreprocessor"
    # Config for global policy
    action_module_cfg.GLOBAL_POLICY = CfgNode()
    action_module_cfg.GLOBAL_POLICY.NAME = "RandomGlobalPolicy"
    action_module_cfg.GLOBAL_POLICY.OBSERVATION_SPACE_SHAPE = [100, 100, 100, 11] # Shape of semantic map patch
    # Config for global policy LR schedule
    action_module_cfg.GLOBAL_POLICY.LR_SCHEDULE = CfgNode()
    action_module_cfg.GLOBAL_POLICY.LR_SCHEDULE.NAME = "ConstantLR"
    action_module_cfg.GLOBAL_POLICY.LR_SCHEDULE.INIT_LR = 0.0001
    # Config for local policy
    action_module_cfg.LOCAL_POLICY = CfgNode()
    action_module_cfg.LOCAL_POLICY.DISTANCE_THRESHOLD = 0.1 #m
    # Config for inference
    action_module_cfg.ACTION_PIPELINE = CfgNode()
    action_module_cfg.ACTION_PIPELINE.IS_DETERMINISTIC = True
    action_module_cfg.ACTION_PIPELINE.GLOBAL_POLICY_POLLING_FREQUENCY = 10
    return action_module_cfg
