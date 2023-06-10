from typing import TypedDict

import numpy as np
import torch
from jaxtyping import Bool, Float32, Int64
from nptyping import NDArray, Shape, UInt8
from yacs.config import CfgNode
from torch.utils.data import DataLoader
from fire import Fire
from tqdm import tqdm



from src.utils.category_mapping import get_scene_index_to_reseal_index_vectorized
from src.data.MaskRCNNEvaluationDataset import MaskRCNNEvaluationDataset, eval_collate_fn
from src.model.perception.data_generator import DataGenerator
from src.model.perception.model_wrapper import ModelWrapper, GroundTruthDict
from src.model.perception.perception_pipeline_config import get_perception_cfg

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
        min_y = np.min(np.where(semantics == object_index)[0])
        min_x = np.min(np.where(semantics == object_index)[1])
        max_y = np.max(np.where(semantics == object_index)[0])
        max_x = np.max(np.where(semantics == object_index)[1])

        boxes.append([min_x, min_y, max_x, max_y])
        labels.append(reseal_semantics[semantics == object_index][0])
        masks.append(semantics == object_index)

    if len(boxes) == 0:
        return {'boxes': torch.zeros((0,4)).float(), 'labels': torch.zeros((0)).long(), 
                'masks': torch.zeros((0, 256, 256)).bool()}

    boxes_tensor = torch.tensor(boxes).float()
    labels_tensor = torch.tensor(labels).long()
    masks_tensors = [torch.tensor(mask).bool() for mask in masks]
    masks_tensor = torch.stack(masks_tensors)
    
    return {'boxes': boxes_tensor, 'labels': labels_tensor, 'masks': masks_tensor}

def evaluate_perception_model(model : ModelWrapper, perception_cfg:CfgNode)->None:
    
    perception_cfg.DATA_GENERATOR.SPLIT = 'minival'
    perception_cfg.DATA_GENERATOR.NUM_SCENES = 3
    data_generator = DataGenerator(perception_cfg, evaluation_mode=True)
    data_generator(model, 0)
    eval_dataset = MaskRCNNEvaluationDataset(perception_cfg.DATA_PATHS, perception_cfg.DATA_GENERATOR.SPLIT, 0)
    eval_dataloader = DataLoader(eval_dataset,
                                    batch_size=perception_cfg.TRAINING.BATCH_SIZE,
                                    shuffle=perception_cfg.TRAINING.SHUFFLE,
                                    num_workers=perception_cfg.TRAINING.NUM_WORKERS,
                                    collate_fn=eval_collate_fn)
    metrics = 0
    count = 0
    for images, semantics, semantic_info_paths in tqdm(eval_dataloader):
        count += 1
        truth_dicts = []
        for i in range(len(semantics)):
            truth_dict = get_ground_truth(semantics[i], semantic_info_paths[i])
            truth_dicts += [truth_dict]
        metrics += model.get_metrics(images, truth_dicts)
    return metrics/count

def eval_perception_model_with_action_policy(weights = None, **kwargs) -> None:
    perception_cfg = get_perception_cfg()
    perception_cfg = load_kwargs_to_config(kwargs, perception_cfg)
        
    # initialize perception model
    model = ModelWrapper(model_config=perception_cfg.MODEL, weights=weights)
    model.cuda()
    
    evaluate_perception_model(model, perception_cfg)

def load_kwargs_to_config(kwargs, config: CfgNode) -> CfgNode:
    for k,v in kwargs.items():
        val = config
        for key in k.split("."):
            key = key.upper()
            if not isinstance(val[key], CfgNode):
                val[key] = v
                break
            val = val[key]
    return config

if __name__ == '__main__':
    Fire(eval_perception_model_with_action_policy)
