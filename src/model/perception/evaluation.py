from typing import TypedDict

import numpy as np
import torch
from jaxtyping import Bool, Float32, Int64
from nptyping import NDArray, Shape, UInt8

from ...utils.category_mapping import get_scene_index_to_reseal_index_vectorized


class GroundTruthDict(TypedDict):
    boxes: Float32[torch.Tensor, "N 4"]
    labels: Int64[torch.Tensor, "N"]
    # The masks are binary
    masks: Bool[torch.Tensor, "N H W"]

def get_ground_truth(semantics: NDArray[Shape["256, 256"], UInt8], semantic_info_file_path: str) -> GroundTruthDict:
    mapping = get_scene_index_to_reseal_index_vectorized(semantic_info_file_path)
    reseal_semantics = mapping(semantics)
    semantics = semantics*(reseal_semantics>0)
    
    boxes = []
    labels = []
    masks = []

    for object_index in np.unique(semantics):
        if object_index == 0:
            continue
        # Find the minimum and maximum index in each dimension where the index is present in the cleared_semantics
        min_x = np.min(np.where(semantics == object_index)[0])
        min_y = np.min(np.where(semantics == object_index)[1])
        max_x = np.max(np.where(semantics == object_index)[0])
        max_y = np.max(np.where(semantics == object_index)[1])
        boxes.append([min_x, min_y, max_x, max_y])
        labels.append(reseal_semantics[semantics == object_index][0])
        masks.append(semantics == object_index)

    boxes_tensor = torch.tensor(boxes).float()
    labels_tensor = torch.tensor(labels).long()
    masks_tensors = [torch.tensor(mask).bool() for mask in masks]
    masks_tensor = torch.stack(masks_tensors)
    
    return {'boxes': boxes_tensor, 'labels': labels_tensor, 'masks': masks_tensor}
