import operator
from typing import Callable

import numpy as np
from scipy.ndimage.measurements import find_objects, label
from scipy.ndimage.morphology import binary_dilation

from ...utils.datatypes import LabelMap3DCategorical, LabelMap3DOneHot, SemanticMap3D


def fill_holes(label_map: LabelMap3DCategorical, number_of_voxels_threshold: int,
               comparator: Callable = operator.le) -> LabelMap3DCategorical:
    # The inverted map will be 1 wherever there is no labeled object. This means that we can use find_objects to find
    # holes and fill them.
    inv_map = label_map[..., 1] == 0

    labeled_hole_map, _ = label(inv_map)
    hole_sizes = np.bincount(labeled_hole_map.flatten())[1:]


    # Find the labels of components smaller than the threshold
    small_holes = np.where(comparator(hole_sizes, number_of_voxels_threshold))[0] + 1
    # Find all bounding boxes
    hole_bounding_boxes = np.array(find_objects(labeled_hole_map))

    for hole_label in small_holes:
        # Find the bounding box of the hole
        bounding_box = hole_bounding_boxes[0]
        # Get the labels of the voxels in the bounding box. Since the hole might be exactly rectangular, we need to
        # first increase the bounding box by 1 in each direction to make sure we get the neighbors of the hole voxels
        bounding_box = [slice(max(0, bounding_box[0].start - 1), min(label_map.shape[0], bounding_box[0].stop + 1)),
                        slice(max(0, bounding_box[1].start - 1), min(label_map.shape[1], bounding_box[1].stop + 1)),
                        slice(max(0, bounding_box[2].start - 1), min(label_map.shape[2], bounding_box[2].stop + 1))]

        bounding_box_labels = label_map[bounding_box[0], bounding_box[1], bounding_box[2], 1].flatten()
        labels_of_labelled_neighbors = bounding_box_labels[bounding_box_labels>0]
        # Calculate the most frequent onehot encoding
        most_frequent_label = np.argmax(np.bincount(labels_of_labelled_neighbors))


        label_map[labeled_hole_map == hole_label, 0] = 1
        # Here, we do not need to add or subract from the most_frequent_label because we have the unlabeled dimension
        label_map[labeled_hole_map == hole_label, 1] = most_frequent_label

    return label_map

def remove_small_objects(label_map: LabelMap3DCategorical, number_of_voxels_threshold: int) -> LabelMap3DCategorical:
    labeled_map, _ = label(label_map[..., 1])
    object_sizes = np.bincount(labeled_map.flatten())[1:]

    small_objects = np.where(object_sizes <= number_of_voxels_threshold)[0] + 1

    for small_object in small_objects:
        label_map[labeled_map == small_object, 1] = 0

    return label_map

def dilate_onehot_label_map(label_map: LabelMap3DOneHot):
    for i in range(1, label_map.shape[-1]):
        label_map[..., i] = binary_dilation(label_map[..., i], iterations=1)
    label_map[label_map[..., 0] == 0, :] = 0
    return label_map

def dilate_map(label_map: LabelMap3DCategorical):
    """ Dilate categorical label map by converting it to onehot, dilating, and converting back to categorical.

    Args:
        label_map (LabelMap3DCategorical): _description_

    Returns:
        _type_: _description_
    """    
    onehot_label_map = categorical_label_map_to_onehot_label_map(label_map)
    onehot_label_map_dilated = dilate_onehot_label_map(onehot_label_map)
    return onehot_label_map_to_categorical_label_map(onehot_label_map_dilated)

# TODO: num_semantic_classes is a very bad name, as it's actually num_semantic_classes+1 for occupancy
def categorical_label_map_to_onehot_label_map(label_map: LabelMap3DCategorical, num_semantic_classes: int = 7
                                              ) -> LabelMap3DOneHot:
    # Extract the occupancy and category arrays
    occupancy = label_map[..., 0]
    category = label_map[..., 1]

    # One-hot encode the category array
    one_hot = np.eye(num_semantic_classes)[category]
    one_hot[..., 0] = occupancy
    return one_hot.astype(bool)

def onehot_label_map_to_categorical_label_map(one_hot_label_map: LabelMap3DOneHot) -> LabelMap3DCategorical:
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
