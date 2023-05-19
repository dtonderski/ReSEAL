import numpy as np
from scipy.ndimage.measurements import find_objects, label
from scipy.ndimage.morphology import binary_dilation
from yacs.config import CfgNode

from src.utils.datatypes import InstanceMap3DCategorical, LabelMap3DCategorical, LabelMap3DOneHot, SemanticMap3D


class MapProcessor:
    """ This class processes a semantic map into a categorical label map, applies multiple morphological operations
    to it and converts it to a categorical instance map. The two maps can be accessed through the properties
    categorical_label_map and categorical_instance_map.
    """
    _config: CfgNode

    @property
    def categorical_label_map(self) -> LabelMap3DCategorical:
        return self._categorical_label_map

    @property
    def categorical_instance_map(self) -> InstanceMap3DCategorical:
        return self._categorical_instance_map


    def __init__(self, semantic_map: SemanticMap3D, map_processor_cfg: CfgNode) -> None:
        """ Initializes a MapProcessor from a semantic map and a config.

        Args:
            semantic_map (SemanticMap3D): the semantic 3d map to convert to a label map.
            map_processor_cfg (CfgNode): config. Must contain: \
                1. NO_OBJECT_CONFIDENCE_THRESHOLD (float): the confidence threshold above which an object is \
                    considered present. \
                2. HOLE_VOXEL_THRESHOLD (int): the number of voxels below which a hole is considered small and is \
                    filled. \
                3. OBJECT_VOXEL_THRESHOLD (int): the number of voxels below which an object is considered small and is \
                    removed.
                4. DILATE (bool): whether to dilate the label map after filling holes and removing small objects.
        """
        self._config = map_processor_cfg
        self._categorical_label_map = self._semantic_map_to_categorical_label_map(semantic_map)
        self._process()
        self._categorical_instance_map = self._categorical_label_map_to_categorical_instance_map(
            self._categorical_label_map)

    def _process(self) -> None:
        """Process the categorical label map by filling holes, removing small objects and dilating if needed.
        """
        self._fill_holes()
        self._remove_small_objects()
        if self._config.DILATE:
            self._dilate_map()

    def _semantic_map_to_categorical_label_map(self, semantic_map: SemanticMap3D) -> LabelMap3DCategorical:
        """Convert a semantic map to a categorical label map by copying the occupancy, thresholding the semantic info \
        of the semantic map, and setting the semantic info of the label map to the maximum category if there is any \
        above the threshold and 0 otherwise.

        Args:
            semantic_map (SemanticMap3D): the semantic map to convert.
            no_object_threshold (float): the threshold above which an object is considered present.

        Returns:
            LabelMap3DCategorical: the categorical label map.
        """
        occupancy = semantic_map[..., 0]
        semantic_info = semantic_map[..., 1:]
        semantic_info[semantic_info < self._config.NO_OBJECT_CONFIDENCE_THRESHOLD] = 0
        # Do onehot encoding. Because of the sum, this will be 0 only if all semantic labels are below threshold.
        label_map = np.argmax(semantic_info, axis = -1) + np.any(semantic_info, axis = -1)
        return np.stack([occupancy, label_map], axis=-1).astype(int)

    def _fill_holes(self) -> None:
        """ Fill holes in the categorical label map. Holes are defined as voxels that are not occupied. Holes smaller \
        than HOLE_VOXEL_THRESHOLD are filled.
        """
        # The inverted map will be 1 wherever there is no labeled object. This means that we can use the scipy.ndimage
        # functions label and find_objects to find holes and fill them.
        inv_map = self._categorical_label_map[..., 1] == 0

        labeled_hole_map, _ = label(inv_map)
        hole_sizes = np.bincount(labeled_hole_map.flatten())[1:]


        # Find the labels of components smaller than the threshold
        small_holes = np.where(hole_sizes < self._config.HOLE_VOXEL_THRESHOLD)[0] + 1
        # Find all bounding boxes
        hole_bounding_boxes = np.array(find_objects(labeled_hole_map))

        for hole_label in small_holes:
            # Find the bounding box of the hole
            bounding_box = hole_bounding_boxes[hole_label-1]
            # Get the labels of the voxels in the bounding box. Since the hole might be exactly rectangular, we need to
            # first increase the bounding box by 1 in each direction to make sure we get the neighbors of the hole
            # voxels
            bounding_box = [slice(max(0, bounding_box[0].start - 1),
                                  min(self._categorical_label_map.shape[0], bounding_box[0].stop + 1)),
                            slice(max(0, bounding_box[1].start - 1),
                                  min(self._categorical_label_map.shape[1], bounding_box[1].stop + 1)),
                            slice(max(0, bounding_box[2].start - 1),
                                  min(self._categorical_label_map.shape[2], bounding_box[2].stop + 1))]

            bounding_box_labels = self._categorical_label_map[
                bounding_box[0], bounding_box[1], bounding_box[2], 1
                ].flatten()
            labels_of_labelled_neighbors = bounding_box_labels[bounding_box_labels>0]
            # Calculate the most frequent onehot encoding
            most_frequent_label = np.argmax(np.bincount(labels_of_labelled_neighbors))


            self._categorical_label_map[labeled_hole_map == hole_label, 0] = 1
            # Here, we do not need to add or subract from the most_frequent_label because we have the unlabeled
            # dimension
            self._categorical_label_map[labeled_hole_map == hole_label, 1] = most_frequent_label

    def _remove_small_objects(self) -> None:
        """ Remove small objects from the categorical label map. Objects are defined as voxels that are occupied and \
        have a semantic label. Objects smaller than OBJECT_VOXEL_THRESHOLD are removed.
        """
        labeled_map, _ = label(self._categorical_label_map[..., 1])
        object_sizes = np.bincount(labeled_map.flatten())[1:]

        small_objects = np.where(object_sizes <= self._config.OBJECT_VOXEL_THRESHOLD)[0] + 1

        for small_object in small_objects:
            self._categorical_label_map[labeled_map == small_object, 1] = 0

    def _dilate_map(self):
        """ Dilate the categorical label map. This is done by first converting it to a onehot label map, dilating it,
        and then converting it back to a categorical label map.
        """
        onehot_label_map = self.categorical_label_map_to_onehot_label_map(self._categorical_label_map)
        onehot_label_map_dilated = self.dilate_onehot_label_map(onehot_label_map)
        self._categorical_label_map = self.onehot_label_map_to_categorical_label_map(onehot_label_map_dilated)

    @staticmethod
    def dilate_onehot_label_map(onehot_label_map: LabelMap3DOneHot) -> LabelMap3DOneHot:
        """ Dilate a onehot label map. This is done by dilating each semantic label separately and then setting the \
            occupancy to 1 wherever there is any semantic label.

        Args:
            onehot_label_map (LabelMap3DOneHot): the onehot label map to dilate.

        Returns:
            LabelMap3DOneHot: the dilated onehot label map.
        """
        for i in range(1, onehot_label_map.shape[-1]):
            onehot_label_map[..., i] = binary_dilation(onehot_label_map[..., i], iterations=1)
            onehot_label_map[onehot_label_map[..., i], 0] = 1
        return onehot_label_map

    @staticmethod
    # TODO: num_semantic_classes is a very bad name, as it's actually num_semantic_classes+1 for occupancy
    def categorical_label_map_to_onehot_label_map(label_map: LabelMap3DCategorical, num_semantic_classes: int = 7
                                                ) -> LabelMap3DOneHot:
        """ Convert a categorical label map to a onehot label map.

        Args:
            label_map (LabelMap3DCategorical): categorical label map to be converted.
            num_semantic_classes (int, optional): Number of semantic classes. Defaults to 6 for ReSEAL.

        Returns:
            LabelMap3DOneHot: the onehot label map.
        """
        # Extract the occupancy and category arrays
        occupancy = label_map[..., 0]
        category = label_map[..., 1]

        # One-hot encode the category array. The occupancy information is stored in the first dimension.
        one_hot = np.eye(num_semantic_classes+1)[category]
        one_hot[..., 0] = occupancy
        return one_hot.astype(bool)

    @staticmethod
    def onehot_label_map_to_categorical_label_map(one_hot_label_map: LabelMap3DOneHot) -> LabelMap3DCategorical:
        """ Convert a onehot label map to a categorical label map.

        Args:
            one_hot_label_map (LabelMap3DOneHot): onehot label map to be converted.

        Returns:
            LabelMap3DCategorical: the categorical label map.
        """
        # Extract the occupancy array from the first dimension of the one_hot_label_map
        occupancy = one_hot_label_map[..., 0]

        # Remove the occupancy information from the one_hot_label_map
        category_one_hot = one_hot_label_map[..., 1:]

        # Convert the one-hot-encoded category array back to the categorical format.
        # Add sum to differentiate between all zeros and the first category.
        category = np.argmax(category_one_hot, axis=-1) + np.sum(category_one_hot, axis=-1)

        # Combine the occupancy and category arrays to form the categorical label map
        categorical_label_map = np.stack((occupancy, category), axis=-1)

        return categorical_label_map

    def _categorical_label_map_to_categorical_instance_map(self, categorical_label_map) -> InstanceMap3DCategorical:
        instance_map_instances, _ = label(categorical_label_map[..., 1])
        # Stack the occupancy information and the instance information
        return np.stack([categorical_label_map[..., 0], instance_map_instances], axis=-1)
