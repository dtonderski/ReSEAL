import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader
from src.config import default_maskrcnn_cfg
from src.data.MaskRCNNDataset import MaskRCNNDataset
from ..utils.datatypes import SemanticMap2D, RGBImage
from ..utils import category_mappings

# Setting up the MaskRCNN considering the config.py
# returns model with corresponing config weights and transforms
def init_maskrcnn():
    cfg = default_maskrcnn_cfg()
    model = None
    weights = None
    if(cfg.MODEL == "resnet50_fpn"):
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        transforms = weights.transforms()
        model = maskrcnn_resnet50_fpn(weights = weights)
    return model, weights, transforms

# Method to generate semantig masks 
# iterating through all scenes and sampled trajectories
# evaluating model for one trajectory at a time
def generating_semantic_masks(perception_model, transformed_images):
    # Needs to include all categories from MaskRCNN (total 81 categories)
    dictionary = {65:0, 62:1, 63:2, 64:3, 70:4, 72:5}

    cfg = default_maskrcnn_cfg()
    #maskrcnn_to_maskcat = category_mappings.load_mask_instance_to_maskcat(cfg)
    
    # dimensions of the SemanticMap2D
    height = transformed_images[0].shape[0]
    width = transformed_images[0].shape[0]
    num_channels = cfg.NUM_CATEGORIES

    train_loader = DataLoader(transformed_images, cfg.BATCHSIZE, cfg.SHUFFLE)
    perception_model.eval()
    semantic_masks_list = []
    
    for _, batch in enumerate(train_loader):
        model_outputs = perception_model(batch)
        semantic_map_2d = torch.zeros((height, width, num_channels))
        for output in model_outputs:
            seal_labels = output['labels'].cpu().apply_(dictionary.get)
            masks = output['masks']
    return semantic_masks_list