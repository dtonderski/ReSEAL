from pathlib import Path
import os
from datetime import datetime

from yacs.config import CfgNode

def get_perception_cfg() -> CfgNode:
    """ Wrapper that includes all necessary configs for perception training.
    """
    perception_cfg = CfgNode()
    perception_cfg.ACTION_MODULE = action_module_cfg()
    perception_cfg.DATA_GENERATOR = data_generator_cfg()
    perception_cfg.DATA_PATHS = data_paths_cfg()
    perception_cfg.MAP_BUILDER = map_builder_cfg()
    perception_cfg.MAP_PROCESSOR = map_processor_cfg()
    perception_cfg.MODEL = model_cfg()
    perception_cfg.SIM = sim_cfg()
    perception_cfg.TRAINING = training_cfg()
    perception_cfg.WANDB = wandb_cfg()
    
    return perception_cfg

def wandb_cfg() -> CfgNode:
    wandb_cfg = CfgNode()
    wandb_cfg.USE_WANDB = False
    wandb_cfg.LOG_MAP = True
    wandb_cfg.LOG_LOSS = True
    wandb_cfg.LOG_SCENE_IDS = True
    return wandb_cfg

def data_generator_cfg() -> CfgNode:
    """ All of these parameters are used during perception training. For a description of the parameters, see the 
        train_perception_model_with_action_policy function in src/model/perception/perception_pipeline.py. The default 
        arguments should be seen as examples.
    """
    data_generator_cfg = CfgNode()
    data_generator_cfg.NUM_SCENES = 10
    data_generator_cfg.NUM_STEPS = 200
    data_generator_cfg.SPLIT = 'train'
    data_generator_cfg.SEED = 0
    data_generator_cfg.NUM_WORKERS = 4 # currently unused
    data_generator_cfg.SEMANTIC_SCENES_ONLY = True
    return data_generator_cfg

def data_paths_cfg():
    """ Here, you might have to modify the DATA_DIR if you have downloaded the data to a different location.
    """
    data_paths_cfg = CfgNode()
    if "SCRATCH" in os.environ:
        data_paths_cfg.DATA_DIR = str(Path(os.environ['SCRATCH'], "3dvis", "ReSEAL", "data"))
        print(f"SCRATCH detected - setting DATA_DIR to {data_paths_cfg.DATA_DIR}")
    else:
        data_paths_cfg.DATA_DIR = str(Path("data"))
    data_paths_cfg.RAW_DATA_DIR = str(Path(data_paths_cfg.DATA_DIR, "raw"))
    data_paths_cfg.INTERIM_DATA_DIR = str(Path(data_paths_cfg.DATA_DIR, "interim"))
    data_paths_cfg.TRAJECTORIES_DIR = str(Path(
        data_paths_cfg.INTERIM_DATA_DIR, "trajectories", datetime.now().strftime("%y-%m-%d,%H:%M:%S")))
    data_paths_cfg.ANNOTATED_SCENE_CONFIG_PATH_IN_SPLIT = str(
        Path("scene_datasets", "hm3d", "hm3d_annotated_basis.scene_dataset_config.json")
    )
    data_paths_cfg.BASIS_SCENE_DATASET_CONFIG_PATH_IN_SPLIT = str(
        Path("scene_datasets", "hm3d", "hm3d_basis.scene_dataset_config.json")
    )
    data_paths_cfg.MODEL_DIR = str(Path("models"))
    return data_paths_cfg

def sim_cfg() -> CfgNode:
    """ This is an exact copy of the defaults at time of writing. Should probably never be modified.
    """
    sim_cfg = CfgNode()
    sim_cfg.SENSOR_CFG = sensor_cfg()
    sim_cfg.FORWARD_MOVE_DISPLACEMENT = 0.25 #m
    sim_cfg.TURN_ANGLE_DISPLACEMENT = 30 #deg
    sim_cfg.DEFAULT_AGENT_ID = 0
    sim_cfg.DEFAULT_POSITION = [-0.6, 0.0, 0.0]  # m
    return sim_cfg

def sensor_cfg() -> CfgNode:
    """ This is an exact copy of the defaults at time of writing. Should probably never be modified.
    """
    sensor_cfg = CfgNode()
    sensor_cfg.WIDTH = 256 #px
    sensor_cfg.HEIGHT = 256 #px
    sensor_cfg.HFOV = 90 #deg
    sensor_cfg.SENSOR_HEIGHT = 0.88 #m
    sensor_cfg.ORTHO_SCALE = 0.1 #m/px
    return sensor_cfg

def map_builder_cfg() -> CfgNode:
    """ This should not be changed. Used in perception training:
        - MAP_BUILDER.RESOLUTION
        - MAP_BUILDER.NUM_SEMANTIC_CLASSES
    """
    map_builder_cfg = CfgNode()
    map_builder_cfg.RESOLUTION = 0.05  # m per pixel
    map_builder_cfg.MAP_SIZE = (25, 1.5, 25)  # (x, y, z) in m
    map_builder_cfg.NUM_SEMANTIC_CLASSES = 6
    return map_builder_cfg

def map_processor_cfg() -> CfgNode:
    """ This is the config that should be changed if you want to change the semantic map preprocessing.
    """
    map_processor_cfg = CfgNode()
    map_processor_cfg.NO_OBJECT_CONFIDENCE_THRESHOLD = 0.5
    map_processor_cfg.HOLE_VOXEL_THRESHOLD = 2000
    map_processor_cfg.OBJECT_VOXEL_THRESHOLD = 200
    map_processor_cfg.DILATE = True
    map_processor_cfg.EXPAND_OCCUPANCY = False
    return map_processor_cfg

def action_module_cfg() -> CfgNode:
    """ Used in perception training:
        - GLOBAL_POLICY.NAME
        - LOCAL_POLICY.DISTANCE_THRESHOLD
        - ACTION_PIPELINE.GLOBAL_POLICY_POLLING_FREQUENCY
        Othe parameters should not be changed.
    """
    action_module_cfg = CfgNode()
    # Config for semantic map preprocessor
    action_module_cfg.PREPROCESSOR = CfgNode()
    action_module_cfg.PREPROCESSOR.NAME = "DummyPreprocessor"
    # Config for global policy
    action_module_cfg.GLOBAL_POLICY = CfgNode()
    action_module_cfg.GLOBAL_POLICY.NAME = "RandomGlobalPolicy"
    action_module_cfg.GLOBAL_POLICY.OBSERVATION_SPACE_SHAPE = [1,1,1,1] # Shape of semantic map patch
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
    action_module_cfg.ACTION_PIPELINE.GLOBAL_POLICY_POLLING_FREQUENCY = 50
    return action_module_cfg

def model_cfg() -> CfgNode:
    """ Main function that might be interesting to modify is the SCORE_THRESHOLD.
    """
    model_cfg = CfgNode()
    model_cfg.USE_INITIAL_TRANSFORMS = True
    model_cfg.SCORE_THRESHOLD = 0.5
    model_cfg.MASK_THRESHOLD = 0.5
    model_cfg.TRAINABLE_BACKBONE_LAYERS = 0
    return model_cfg

def training_cfg() -> CfgNode:
    """ This defines the training parameters.
    """
    train_cfg = CfgNode()
    train_cfg.NUM_EPOCHS = 30
    train_cfg.NUM_CLASSES = 6
    train_cfg.BATCH_SIZE = 4
    train_cfg.SHUFFLE = True
    train_cfg.NUM_WORKERS = 4
    train_cfg.LEARNING_RATE = 0.005
    train_cfg.OPTIM_MOMENTUM = 0.9
    train_cfg.OPTIM_WEIGHT_DECAY = 0.0005
    train_cfg.OPTIM_STEP_SIZE = 3
    train_cfg.OPTIM_GAMMA = 0.1    
    return train_cfg