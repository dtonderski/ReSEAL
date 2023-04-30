from typing import Callable

import numpy as np
from nptyping import Int, NDArray, Shape
from src.utils.datatypes import SemanticMap3D


def sorted_dict_by_value(dictionary):
    return dict(sorted(dictionary.items(), key=lambda item: item[1]))

def get_semantic_map(
        saved_semantics: NDArray[Shape["Height, Width"], Int],
        scene_index_to_category_index_map: Callable,
        num_semantic_classes: int
    ) -> NDArray[Shape["Height, Width, NumSemanticClasses"], Int]:
    """ Function to get semantic map from saved semantics. It first converts the saved semantics to the category \
        indices by running the scene_index_to_category_index_map function on the saved semantics. Then, it converts to \
        a one-hot encoding and discards the channel corresponding to index 0. In matterport, this is void, in reseal, \
        it is unlabeled, and in maskrcnn, it isn't used.

    Args:
        saved_semantics (NDArray[Shape["Height, Width"], Int]): saved semantics from the simulator.
        scene_index_to_category_index_map (Callable): function to convert scene indices to category indices.
        num_semantic_classes (int): number of semantic classes in the new category mapping.

    Returns:
        NDArray[Shape["Height, Width, NumSemanticClasses"], Int]: one-hot encoded semantic map using the new category \
            mapping.
    """
    semantic_map_reseal_indices = scene_index_to_category_index_map(saved_semantics)
    semantic_map = np.zeros((saved_semantics.size, num_semantic_classes + 1))
    semantic_map[np.arange(semantic_map_reseal_indices.size), semantic_map_reseal_indices.flatten()] = 1
    semantic_map = semantic_map.reshape(saved_semantics.shape + (num_semantic_classes + 1,))[:,:,1:]
    return semantic_map

def semantic_map_to_categorical_map(
        semantic_map: SemanticMap3D
    ) -> NDArray[Shape["NumPixelsX, NumPixelsY, NumPixelsZ"], Int]:
    """ Function to extract the categories for a semantic map. If a voxel has no semantic information, it gets category
        0, otherwise, it gets a category between 1 and NUM_SEMANTIC_CLASSES.

    Args:
        semantic_map (SemanticMap3D): one-hot encoded semantic map.

    Returns:
        NDArray[Shape["Height, Width"], Int]: semantic categories of the semantic map.
    """
    return (semantic_map[:,:,:,1:].argmax(axis=-1) + semantic_map[:,:,:,1:].sum(axis=-1))
