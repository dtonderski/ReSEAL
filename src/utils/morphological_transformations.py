from scipy.ndimage.measurements import label, find_objects
import numpy as np
from ..utils.datatypes import LabelMap3D, SemanticMap3D
import operator
from typing import Callable

def semantic_map_to_label_map(semantic_map: SemanticMap3D, no_object_threshold: float) -> LabelMap3D:
    semantic_info = semantic_map[:, :, :, 1:]
    semantic_info[semantic_info < no_object_threshold] = 0
    # Do onehot encoding. Because of the sum, this will be 0 only if all semantic labels are below threshold.
    label_map = np.argmax(semantic_info, axis = -1) + np.any(semantic_info, axis = -1) 
    return label_map

def fill_holes(label_map: LabelMap3D, number_of_voxels_threshold: int,
               comparator: Callable = operator.le) -> LabelMap3D:
    # The inverted map will be 1 wherever there is no semantic object. This means that we can use find_objects to find 
    # holes and fill them.
    inv_map = label_map == 0

    labeled_hole_map, _ = label(inv_map)
    hole_sizes = np.bincount(labeled_hole_map.flatten())[1:]

    # Find the labels of components smaller than the threshold
    small_holes = np.where(comparator(hole_sizes, number_of_voxels_threshold))[0] + 1

    for hole_label in small_holes:
        # Find the bounding box of the hole
        bounding_box = find_objects(labeled_hole_map == hole_label)[0]
        # Get the labels of the voxels in the bounding box. Since the hole might be exactly rectangular, we need to
        # first increase the bounding box by 1 in each direction to make sure we get the neighbors of the hole voxels
        bounding_box = [slice(max(0, bounding_box[0].start - 1), min(label_map.shape[0], bounding_box[0].stop + 1)),
                        slice(max(0, bounding_box[1].start - 1), min(label_map.shape[1], bounding_box[1].stop + 1)),
                        slice(max(0, bounding_box[2].start - 1), min(label_map.shape[2], bounding_box[2].stop + 1))]

        bounding_box_labels = label_map[bounding_box[0], bounding_box[1], bounding_box[2]]

        # Calculate the most frequent onehot encoding
        most_frequent_label = np.argmax(np.bincount(bounding_box_labels))

        if most_frequent_label > 0:
            # Here, we do not need to subtract 1 from the most_frequent_label because we have the occupancy channel
            label_map[labeled_hole_map == hole_label] = most_frequent_label
