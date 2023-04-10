import json
from typing import Tuple
from pathlib import Path

import numpy as np
import habitat_sim
from yacs.config import CfgNode

from ..config import default_data_paths_cfg, default_sim_cfg, default_sensor_cfg

def initialize_sim(scene_split: str,
                   scene_id: str,
                   data_paths_cfg: CfgNode = None,
                   sim_cfg: CfgNode = None,
                   verbose:bool=False) -> habitat_sim.Simulator:
    """ Initialize the simulator from a given scene

    Args:
        scene_split (str): 'train', 'val', or 'minival'.
        scene_id (str): string of the form [0-9]{5}-[0-z]{11}, for example 00800-TEEsavR23oF.
        data_paths_cfg (CfgNode, optional): data path yacs config. Defaults to None. \
            If None, uses default_data_paths_cfg()
        sim_cfg (CfgNode, optional): sim yacs config. Defaults to None. If None, uses default_sim_cfg().
        verbose (bool, optional): if set, prints agent position and rotation after init. Defaults to False.

    Returns:
        habitat_sim.Simulator: simulator with the current scene loaded. The agent will have a semantic sensor if \
            scene is part of HM3D-SEM.
    """
    if sim_cfg is None:
        sim_cfg = default_sim_cfg()

    habitat_sim_cfg = make_habitat_sim_cfg(scene_split, scene_id, data_paths_cfg, sim_cfg)
    sim = habitat_sim.Simulator(habitat_sim_cfg)

    # Set agent state
    agent = sim.initialize_agent(sim_cfg.DEFAULT_AGENT_ID)
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array(sim_cfg.DEFAULT_POSITION)  # world space
    agent.set_state(agent_state)

    # Get agent state
    agent_state = agent.get_state()
    if verbose:
        print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

    return sim

def check_if_semantic_sensor_used(sim: habitat_sim.Simulator) -> bool:
    """ Utility function that checks whether the first (and usually only) agent in the scene has a semantic sensor.

    Args:
        sim (habitat_sim.Simulator): the initialized simulator

    Returns:
        bool: True if the first agent has a semantic sensor, False otherwise.
    """
    return any(x.sensor_type == habitat_sim.SensorType.SEMANTIC
                for x in sim.config.agents[0].sensor_specifications)


def make_habitat_sim_cfg(scene_split: str,
                         scene_id: str,
                         data_paths_cfg: CfgNode = None,
                         sim_cfg: CfgNode = None) -> habitat_sim.Configuration:
    """ Creates a habitat_sim.Configuration object for a given scene

    Args:
        scene_split (str): 'train', 'val', or 'minival'.
        scene_id (str): string of the form [0-9]{5}-[0-z]{11}, for example 00800-TEEsavR23oF.
        data_paths_cfg (CfgNode, optional): data path yacs config. Defaults to None. \
            If None, uses default_data_paths_cfg()
        sim_cfg (CfgNode, optional): sim yacs config. Defaults to None. If None, uses default_sim_cfg().

    Returns:
        habitat_sim.Configuration: the configuration object for the given scene.
    """
    if sim_cfg is None:
        sim_cfg = default_sim_cfg()

    scene_path, scene_dataset_config_path, use_semantic_sensor = get_scene_info(scene_split, scene_id, data_paths_cfg)


    habitat_sim_cfg = habitat_sim.SimulatorConfiguration()
    habitat_sim_cfg.gpu_device_id = 0
    habitat_sim_cfg.scene_id = str(scene_path)
    habitat_sim_cfg.scene_dataset_config_file = str(scene_dataset_config_path)
    habitat_sim_cfg.enable_physics = False

    # Note: all sensors must have the same resolution
    sensor_specs = []
    sensor_specs.append(get_sensor_spec("color_sensor", sim_cfg.SENSOR_CFG))
    sensor_specs.append(get_sensor_spec("depth_sensor", sim_cfg.SENSOR_CFG))

    if use_semantic_sensor:
        sensor_specs.append(get_sensor_spec("semantic_sensor", sim_cfg.SENSOR_CFG))

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=sim_cfg.FORWARD_MOVE_DISPLACEMENT)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=sim_cfg.TURN_ANGLE_DISPLACEMENT)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=sim_cfg.TURN_ANGLE_DISPLACEMENT)
        ),
    }

    return habitat_sim.Configuration(habitat_sim_cfg, [agent_cfg])


def get_scene_info(scene_split: str,
                   scene_id: str,
                   data_paths_cfg: CfgNode = None) -> Tuple[Path, Path, bool]:
    """Gets the scene path, scene dataset config path, and whether the scene uses a semantic sensor.

    Args:
        scene_split (str): 'train', 'val', or 'minival'.
        scene_id (str): string of the form [0-9]{5}-[0-z]{11}, for example 00800-TEEsavR23oF.
        data_paths_cfg (CfgNode, optional): data path yacs config. Defaults to None. \
            If None, uses default_data_paths_cfg()

    Returns:
        Tuple[Path, Path, bool]: the path to the .glb file, the path to the .json scene dataset config file, and \
            a boolean indicating whether the scene supports a semantic sensor.
    """

    if data_paths_cfg is None:
        data_paths_cfg = default_data_paths_cfg()

    annotated_scene_config_path: Path = Path(data_paths_cfg.RAW_DATA_DIR, scene_split,
                                            data_paths_cfg.ANNOTATED_SCENE_CONFIG_PATH_IN_SPLIT)

    with annotated_scene_config_path.open(encoding='utf-8') as file:
        annotated_scene_config = json.load(file)

    annotated_scene_set = {Path(x).parents[0] for x in annotated_scene_config["stages"]["paths"][".glb"]}
    use_semantic_sensor = Path(scene_split, scene_id) in annotated_scene_set

    scene_id_without_index = scene_id.split('-')[1]
    if use_semantic_sensor:
        scene_dataset_config_path = Path(data_paths_cfg.RAW_DATA_DIR, scene_split,
                                        data_paths_cfg.ANNOTATED_SCENE_CONFIG_PATH_IN_SPLIT)
    else:
        scene_dataset_config_path = Path(data_paths_cfg.RAW_DATA_DIR, scene_split,
                                        data_paths_cfg.BASIS_SCENE_DATASET_CONFIG_PATH_IN_SPLIT)

    scene_filename = f'{scene_id_without_index}.basis.glb'

    scene_path = Path(data_paths_cfg.RAW_DATA_DIR, scene_split,
                    "scene_datasets", "hm3d", scene_split, scene_id, scene_filename)

    return scene_path, scene_dataset_config_path, use_semantic_sensor


def get_sensor_spec(sensor_type: str, sensor_cfg: CfgNode = None) -> habitat_sim.CameraSensorSpec:
    """ Creates a sensor spec for the given sensor type.

    Args:
        sensor_type (str): "color_sensor", "depth_sensor", or "semantic_sensor".
        sim_cfg (CfgNode, optional): yacs simulator configuration. Defaults to None. If None, default_sim_cfg() is used.

    Returns:
        habitat_sim.CameraSensorSpec: sensor spec for the given sensor type.
    """

    assert sensor_type in ["color_sensor", "depth_sensor", "semantic_sensor"]

    if sensor_cfg is None:
        sensor_cfg = default_sensor_cfg()

    sensor_spec: habitat_sim.CameraSensorSpec = habitat_sim.CameraSensorSpec()

    if sensor_type == "color_sensor":
        sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    elif sensor_type == "depth_sensor":
        sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    else:
        sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC

    sensor_spec.uuid = sensor_type
    sensor_spec.resolution = [sensor_cfg.HEIGHT, sensor_cfg.WIDTH]
    sensor_spec.hfov = sensor_cfg.HFOV
    sensor_spec.position = [0.0, sensor_cfg.SENSOR_HEIGHT, 0.0]
    sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    return sensor_spec
