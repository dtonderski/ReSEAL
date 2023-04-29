import numpy as np

class SemanticMap3DBuilderOptimized:
    def __init__(self, map_builder_cfg: CfgNode, sensor_cfg: CfgNode) -> None:
        self._resolution = map_builder_cfg.RESOLUTION  # m per voxel
        self._map_size = np.array(map_builder_cfg.MAP_SIZE)  # map (voxel grid) size in meters
        self._num_semantic_classes = map_builder_cfg.NUM_SEMANTIC_CLASSES
        self._intrinsic = get_camera_intrinsic_from_cfg(sensor_cfg)