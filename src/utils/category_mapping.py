import csv
from typing import Callable, Dict

import numpy as np
from matplotlib.colors import to_rgb
from nptyping import NDArray, Shape, Float
from yacs.config import CfgNode

from ..config import default_data_paths_cfg
from ..utils.misc import sorted_dict_by_value


def get_hm3dsem_raw_names_to_matterport_names_mapping_dict(data_paths_cfg: CfgNode = None) -> Dict[str, str]:
    """ Returns a map from raw hm3dsem category names to matterport category names.
    """
    hm3dsem_mapping = {}
    data_paths_cfg = default_data_paths_cfg() if data_paths_cfg is None else data_paths_cfg

    # Adding this as we need to somehow map pixels which are 0 to matterport 0, which is void
    hm3dsem_mapping['void'] = 'void'

    with open(data_paths_cfg.HM3DSEM_MAPPING_PATH, encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter = '\t')
        next(iter(reader))
        for row in reader:
            hm3dsem_mapping[row[0]] = row[2]
    return hm3dsem_mapping

def get_scene_index_to_matterport_name_map(semantic_info_file_path, data_paths_cfg: CfgNode = None) -> Dict[int, str]:
    hm3dsem_to_matterport_mapping = get_hm3dsem_raw_names_to_matterport_names_mapping_dict(data_paths_cfg)

    with open(semantic_info_file_path, encoding='utf8') as semantic_info_file:
        lines = semantic_info_file.readlines()
        name_dict = {}

        # 0s are void
        name_dict[0] = 'void'
        for line in lines[1:]:
            index, _, name, _ = line.split(',')
            name = name.replace('"', '')
            name_dict[int(index)] = name

    scene_index_to_matterport_name = {k: hm3dsem_to_matterport_mapping[v] for k, v in name_dict.items()}
    return sorted_dict_by_value(scene_index_to_matterport_name)

def get_scene_index_to_matterport_index_map(semantic_info_file_path, data_paths_cfg: CfgNode = None) -> Dict[int, int]:
    scene_index_to_matterport_name = get_scene_index_to_matterport_name_map(semantic_info_file_path, data_paths_cfg)
    data_paths_cfg = default_data_paths_cfg() if data_paths_cfg is None else data_paths_cfg

    matterport_mapping = {}
    with open(data_paths_cfg.MATTERPORT_MAPPING_PATH, encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter = '\t')
        next(iter(reader))
        for row in reader:
            matterport_mapping[row[1]] = int(row[0])
    return {k: matterport_mapping[v] for k, v in scene_index_to_matterport_name.items()}
    

def get_matterport_name_to_reseal_name_map(data_paths_cfg:CfgNode = None) -> Dict[str, str]:
    data_paths_cfg = default_data_paths_cfg() if data_paths_cfg is None else data_paths_cfg

    matterport_to_reseal_mapping = {}
    with open(data_paths_cfg.MATTERPORT_TO_RESEAL_MAPPING_PATH, encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter = '\t')
        next(iter(reader))
        for row in reader:
            matterport_to_reseal_mapping[row[0]] = row[1]
    return matterport_to_reseal_mapping

def get_reseal_name_to_reseal_index_map(data_paths_cfg: CfgNode=None) -> Dict[str, int]:
    data_paths_cfg = default_data_paths_cfg() if data_paths_cfg is None else data_paths_cfg

    reseal_name_to_reseal_index = {}
    with open(data_paths_cfg.RESEAL_MAPPING_PATH, encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter = '\t')
        next(iter(reader))
        for row in reader:
            reseal_name_to_reseal_index[row[1]] = int(row[0])
    return reseal_name_to_reseal_index

def get_matterport_name_to_reseal_index_map(data_paths_cfg = None) -> Dict[str, int]:
    matterport_name_to_reseal_name = get_matterport_name_to_reseal_name_map(data_paths_cfg)
    reseal_name_to_reseal_index = get_reseal_name_to_reseal_index_map(data_paths_cfg)
    return {k: reseal_name_to_reseal_index[v] for k, v in matterport_name_to_reseal_name.items()}

def get_scene_index_to_reseal_index_map(semantic_info_file_path: str, data_paths_cfg:CfgNode = None) -> Dict[int, int]:
    scene_to_matterport_names = get_scene_index_to_matterport_name_map(semantic_info_file_path, data_paths_cfg)
    matterport_name_to_reseal_index = get_matterport_name_to_reseal_index_map(data_paths_cfg)
    return {k: matterport_name_to_reseal_index[v] for k, v in scene_to_matterport_names.items()}

def get_scene_index_to_reseal_index_map_vectorized(semantic_info_file_path, data_paths_cfg:CfgNode = None) -> Callable:
    return np.vectorize(get_scene_index_to_reseal_index_map(semantic_info_file_path, data_paths_cfg).get)

def get_reseal_color_dict(data_paths_cfg: CfgNode = None) -> Dict[int, NDArray[Shape["3"], Float]]:
    """ Get a dictionary mapping from reseal index to reseal color.

    Args:
        data_paths_cfg (CfgNode, optional): optional data_paths configuration. If none, uses default_data_paths_cfg().

    Returns:
        Dict[str, Tuple[int, int, int]]: dictionary mapping from reseal index to reseal color
    """
    data_paths_cfg = default_data_paths_cfg() if data_paths_cfg is None else data_paths_cfg

    reseal_index_to_color = {}
    with open(data_paths_cfg.RESEAL_MAPPING_PATH, encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter = '\t')
        next(iter(reader))
        for row in reader:
            reseal_index_to_color[int(row[0])] = np.array(to_rgb(f"#{row[2]}"))
    return reseal_index_to_color

def get_reseal_color_converter(data_paths_cfg: CfgNode=None) -> Callable:
    """ Get a numpy vectorized dictionary mapping from reseal index to reseal color. Can be chained with\
        get_scene_index_to_reseal_index_map_vectorized to get the color for each pixel in a scene.

    Args:
        data_paths_cfg (CfgNode, optional): optional data_paths configuration.

    Returns:
        Callable: numpy vectorized dictionary mapping from reseal index to reseal color.
    """
    reseal_index_to_color = get_reseal_color_dict(data_paths_cfg)
    return np.vectorize(reseal_index_to_color.get, signature='()->(n)')
