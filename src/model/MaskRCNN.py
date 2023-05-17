import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader
from src.config import default_maskrcnn_cfg
from src.data.MaskRCNNDataset import MaskRCNNDataset
from src.utils.datatypes import SemanticMap2D, RGBImage
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# Setting up the MaskRCNN considering the config.py
# returns model with corresponing config weights and transforms
def build_maskrcnn(num_classes):
    model = maskrcnn_resnet50_fpn(weights = MaskRCNN_ResNet50_FPN_Weights)
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # hidden_layer = 256
    # model.roi_heads.maks_predictor = MaskRCNNPredictor(in_features_mask,
    #                                                        hidden_layer,
    #                                                        num_classes)
    return model

# Method to generate semantig masks 
# iterating through all scenes and sampled trajectories
# evaluating model for one trajectory at a time
def generating_semantic_masks(perception_model, image):
    
    return 