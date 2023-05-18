import numpy as np
import torch
from nptyping import Int, NDArray, Shape
from scipy.ndimage import find_objects
from yacs.config import CfgNode

from src.features.perception import propagate_labels
from src.model.perception.map_processing import MapProcessor
from src.model.perception.model_wrapper import LabelDict
from src.utils.category_mapping import get_instance_index_to_reseal_index_dict
from src.utils.datatypes import InstanceMap3DCategorical, LabelMap3DCategorical, Pose, SemanticMap2D, SemanticMap3D


class LabelGenerator:
    """ This class will generate maskrcnn labels out of SemanticMap3D, grid_index_of_origin, and poses. It also \
        functionality to generate 2d instance maps, which can be useful for visualization. It uses the MapProcessor.
        
        Args:
            semantic_map (SemanticMap3D): the base semantic map to process.
            grid_index_of_origin (NDArray[Shape["3"], Int]): grid index of the origin of the coordinates system.
            map_builder_cfg (CfgNode): config of the map builder. Must include RESOLUTION (float).
            map_processor_cfg (CfgNode): config of the map processor. Must include NO_OBJECT_CONFIDENCE_THRESHOLD \
                (float), HOLE_VOXEL_THRESHOLD (int), OBJECT_VOXEL_THRESHOLD (int), and DILATE (bool).
            sensor_cfg (CfgNode): config of the sensor. Must include HEIGHT (int), WIDTH (int), and FOV (float).
    """
    @property
    def categorical_label_map(self) -> LabelMap3DCategorical:
        return self._map_processor.categorical_label_map

    @property
    def categorical_instance_map(self) -> InstanceMap3DCategorical:
        return self._map_processor.categorical_instance_map

    def __init__(self, semantic_map: SemanticMap3D, grid_index_of_origin: NDArray[Shape["3"], Int],
                 map_builder_cfg: CfgNode, map_processor_cfg: CfgNode, sensor_cfg: CfgNode):
        self._semantic_map = semantic_map
        self._grid_index_of_origin = grid_index_of_origin
        self._map_builder_cfg = map_builder_cfg
        self._map_processor_cfg = map_processor_cfg
        self._sensor_cfg = sensor_cfg
        self._map_processor = MapProcessor(self._semantic_map, self._map_processor_cfg)

    # TODO: this shouldn't be a __call__
    def get_label_dict(self, pose: Pose) -> LabelDict:
        instance_map_2d = self.get_instance_map_2d(pose)
        return self.get_label_dict_from_instance_map(instance_map_2d)

    def get_instance_map_2d(self, pose: Pose):
        sensor_position, sensor_rotation = pose
        return propagate_labels(
            sensor_rotation,
            sensor_position,
            self.categorical_instance_map,
            self._grid_index_of_origin,
            self._map_builder_cfg,
            self._sensor_cfg
        )

    def get_label_dict_from_instance_map(self, instance_map_2d: SemanticMap2D) -> LabelDict:
        # One hot encode the instance map
        one_hot = np.eye(self.categorical_instance_map.max()+1)[instance_map_2d][..., 0, 1:].astype(bool)
        # Get dict mapping instance index to reseal index
        label_dict = get_instance_index_to_reseal_index_dict(self.categorical_instance_map,
                                                             self.categorical_label_map)
        # Get the labels for each instance
        instance_labels = np.array(list(label_dict.values()))[1:]
        # Get the bounding boxes for each instance by using find_objects in the instance map. Note that
        # this works even if there is an instance that consists of two separate parts in the 2d instance map because
        # of how find_objects works.
        bounding_boxes = [self._object_slice_to_bounding_box(object_slice) if object_slice else None
                        for object_slice in find_objects(instance_map_2d)]

        present_instance_indices_mask = (one_hot.sum(axis = (0,1)) > 0)

        boxes = np.array([bounding_box for bounding_box in bounding_boxes if bounding_box is not None])
        labels = instance_labels[present_instance_indices_mask]
        masks = one_hot[..., present_instance_indices_mask].transpose(2,0,1)
        # TODO: Sanity check, remove when sane
        assert boxes.shape[0] == labels.shape[0] == masks.shape[0]
        if boxes.size == 0:
            boxes = np.zeros((0, 4))
            labels = np.zeros((0,))
            masks = np.zeros((0, self._sensor_cfg.HEIGHT, self._sensor_cfg.WIDTH))
        return {"boxes": torch.tensor(boxes).float(),
                "labels": torch.tensor(labels).type(torch.int64),
                "masks": torch.tensor(masks).type(torch.uint8)}

    def _object_slice_to_bounding_box(self, object_slice):
        return np.array([object_slice[1].start, object_slice[0].start, object_slice[1].stop, object_slice[0].stop])
