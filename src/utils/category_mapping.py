import csv
from typing import Callable, Dict, List

import numpy as np
from matplotlib.colors import to_rgb
from nptyping import NDArray, Shape, Float
from yacs.config import CfgNode

from ..config import default_data_paths_cfg
from ..utils.misc import sorted_dict_by_value
from ..utils.datatypes import LabelMap3DCategorical, InstanceMap3DCategorical
from collections import defaultdict

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

def get_scene_index_to_matterport_name_dict(semantic_info_file_path, data_paths_cfg: CfgNode = None) -> Dict[int, str]:
    hm3dsem_name_to_matterport_name = get_hm3dsem_raw_names_to_matterport_names_mapping_dict(data_paths_cfg)

    with open(semantic_info_file_path, encoding='utf8') as semantic_info_file:
        lines = semantic_info_file.readlines()
        scene_index_to_hm3dsem_name = {}

        # 0s are void
        scene_index_to_hm3dsem_name[0] = 'void'
        for line in lines[1:]:
            index, _, name, _ = line.split(',')
            name = name.replace('"', '')
            scene_index_to_hm3dsem_name[int(index)] = name

    return sorted_dict_by_value(chain_dicts(scene_index_to_hm3dsem_name, hm3dsem_name_to_matterport_name))

def get_scene_index_to_matterport_index_dict(semantic_info_file_path, data_paths_cfg: CfgNode = None) -> Dict[int, int]:
    scene_index_to_matterport_name = get_scene_index_to_matterport_name_dict(semantic_info_file_path, data_paths_cfg)
    data_paths_cfg = default_data_paths_cfg() if data_paths_cfg is None else data_paths_cfg

    matterport_name_to_matterport_index = {}
    with open(data_paths_cfg.MATTERPORT_MAPPING_PATH, encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter = '\t')
        next(iter(reader))
        for row in reader:
            matterport_name_to_matterport_index[row[1]] = int(row[0])
    return chain_dicts(scene_index_to_matterport_name, matterport_name_to_matterport_index)

def get_matterport_name_to_reseal_name_dict(data_paths_cfg:CfgNode = None) -> Dict[str, str]:
    data_paths_cfg = default_data_paths_cfg() if data_paths_cfg is None else data_paths_cfg

    matterport_to_reseal_mapping = {}
    with open(data_paths_cfg.MATTERPORT_TO_RESEAL_MAPPING_PATH, encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter = '\t')
        next(iter(reader))
        for row in reader:
            matterport_to_reseal_mapping[row[0]] = row[1]
    return matterport_to_reseal_mapping

def get_reseal_name_to_reseal_index_dict(data_paths_cfg: CfgNode=None) -> Dict[str, int]:
    data_paths_cfg = default_data_paths_cfg() if data_paths_cfg is None else data_paths_cfg

    reseal_name_to_reseal_index = {}
    with open(data_paths_cfg.RESEAL_MAPPING_PATH, encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter = '\t')
        next(iter(reader))
        for row in reader:
            reseal_name_to_reseal_index[row[1]] = int(row[0])
    return reseal_name_to_reseal_index

def get_reseal_index_to_reseal_name_dict(data_paths_cfg: CfgNode=None) -> Dict[int, str]:
    data_paths_cfg = default_data_paths_cfg() if data_paths_cfg is None else data_paths_cfg

    reseal_index_to_reseal_name = {}
    with open(data_paths_cfg.RESEAL_MAPPING_PATH, encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter = '\t')
        next(iter(reader))
        for row in reader:
            reseal_index_to_reseal_name[int(row[0])] = row[1]
    return reseal_index_to_reseal_name

def get_matterport_name_to_reseal_index_dict(data_paths_cfg = None) -> Dict[str, int]:
    return chain_dicts(get_matterport_name_to_reseal_name_dict(data_paths_cfg),
                       get_reseal_name_to_reseal_index_dict(data_paths_cfg))

def get_scene_index_to_reseal_index_dict(semantic_info_file_path: str, data_paths_cfg:CfgNode = None) -> Dict[int, int]:
    return chain_dicts(get_scene_index_to_matterport_name_dict(semantic_info_file_path, data_paths_cfg),
                       get_matterport_name_to_reseal_index_dict(data_paths_cfg))

def get_scene_index_to_reseal_index_vectorized(semantic_info_file_path, data_paths_cfg:CfgNode = None) -> Callable:
    return np.vectorize(get_scene_index_to_reseal_index_dict(semantic_info_file_path, data_paths_cfg).get)

def get_reseal_index_to_color_dict(data_paths_cfg: CfgNode = None) -> Dict[int, NDArray[Shape["3"], Float]]:
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

def get_reseal_index_to_color_vectorized(data_paths_cfg: CfgNode=None) -> Callable:
    """ Get a numpy vectorized dictionary mapping from reseal index to reseal color. Can be chained with\
        get_scene_index_to_reseal_index_map_vectorized to get the color for each pixel in a scene.

    Args:
        data_paths_cfg (CfgNode, optional): optional data_paths configuration.

    Returns:
        Callable: numpy vectorized dictionary mapping from reseal index to reseal color.
    """
    reseal_index_to_color = get_reseal_index_to_color_dict(data_paths_cfg)
    return np.vectorize(reseal_index_to_color.get, signature='()->(n)')

def get_instance_index_to_reseal_index_dict(instance_map: InstanceMap3DCategorical,
                               label_map: LabelMap3DCategorical) -> Dict[int, int]:
    instance_to_reseal_dict = {}
    # Map 0 to 0
    instance_to_reseal_dict[0] = 0
    # For each unique instance number, get the location of the first pixel of that instance, and then map the instance
    # number to the corresponding reseal index given by the value of the label map at that location.
    for instance_number in np.unique(instance_map[..., 1])[1:]:
        first_location_of_instance = tuple(x[0] for x in np.where(instance_map[..., 1] == instance_number))
        instance_to_reseal_dict[instance_number] = label_map[(*first_location_of_instance, 1)]
    return instance_to_reseal_dict

def get_instance_index_to_reseal_name_dict(instance_map: InstanceMap3DCategorical,
                                           label_map: LabelMap3DCategorical,
                                           data_paths_cfg = None) -> Dict[int, str]:
    return chain_dicts(get_instance_index_to_reseal_index_dict(instance_map, label_map),
                       get_reseal_index_to_reseal_name_dict(data_paths_cfg))

def get_maskrcnn_index_to_reseal_name_dict(data_paths_cfg = None) -> Dict[int, str]:
    data_paths_cfg = default_data_paths_cfg() if data_paths_cfg is None else data_paths_cfg
    with open(data_paths_cfg.MASKRCNN_TO_RESEAL_MAPPING_PATH, encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter = '\t')
        next(iter(reader))
        maskrcnn_index_to_reseal_name = {int(row[0]): row[2] for row in reader}
    return maskrcnn_index_to_reseal_name

def get_maskrcnn_index_to_reseal_index_dict(data_paths_cfg = None) -> Dict[int, int]:
    return chain_dicts(get_maskrcnn_index_to_reseal_name_dict(data_paths_cfg),
                       get_reseal_name_to_reseal_index_dict(data_paths_cfg))

def get_reseal_name_to_maskrcnn_index_dict(data_paths_cfg = None) -> Dict[str, int]:
    data_paths_cfg = default_data_paths_cfg() if data_paths_cfg is None else data_paths_cfg
    reseal_name_to_maskrcnn_indices_dict: Dict[str, List[int]] = defaultdict(list)
    with open(data_paths_cfg.MASKRCNN_TO_RESEAL_MAPPING_PATH, encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter = '\t')
        next(iter(reader))
        for row in reader:
            reseal_name_to_maskrcnn_indices_dict[row[2]].append(int(row[0]))
    reseal_name_to_maskrcnn_index_dict = {k: v[0] for k, v in reseal_name_to_maskrcnn_indices_dict.items()}
    # Override the background index
    reseal_name_to_maskrcnn_index_dict["unlabeled"] = 0
    return reseal_name_to_maskrcnn_index_dict

def get_reseal_index_to_maskrcnn_index_dict(data_paths_cfg = None) -> Dict[int, int]:
    return chain_dicts(get_reseal_index_to_reseal_name_dict(data_paths_cfg),
                       get_reseal_name_to_maskrcnn_index_dict(data_paths_cfg))

def chain_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """ Chain two dictionaries together, where the keys of dict1 are mapped to the values of dict2.

    Args:
        dict1 (Dict): first dictionary
        dict2 (Dict): second dictionary

    Returns:
        Dict: chained dictionary
    """
    return {k: dict2[v] for k, v in dict1.items()}
