import numpy as np

from ...utils.datatypes import LabelMap3DCategorical, LabelMap3DOneHot


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
