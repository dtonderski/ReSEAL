import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader
from src.config import default_maskrcnn_cfg
from src.data.MaskRCNNDataset import MaskRCNNDataset
from src.utils.datatypes import SemanticMap2D, RGBImage
from src.utils.category_mapping import get_maskrcnn_to_reseal_map
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
    # Needs to include all categories from MaskRCNN (total 81 categories)
    mask_to_reseal_map = get_maskrcnn_to_reseal_map()
    transform = MaskRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()
    transformed_image = [transform(image)]
    cfg = default_maskrcnn_cfg()
    #maskrcnn_to_maskcat = category_mappings.load_mask_instance_to_maskcat(cfg)
    
    # dimensions of the SemanticMap2D
    height = image.shape[1]
    width = image.shape[2]
    num_channels = cfg.NUM_CLASSES
    
    semantic_map_2d = torch.zeros(height, width, num_channels+1)

    #train_loader = DataLoader(transformed_images, cfg.BATCHSIZE, cfg.SHUFFLE)
    perception_model.eval()

    model_output = perception_model(transformed_image)[0]
    labels = model_output['labels'].cpu()
    for i in range(labels.shape[0]):
        labels[i] = mask_to_reseal_map.get(str(labels[i].item()))
    #labels = model_output['labels'].cpu().apply_(mask_to_reseal_map.get)
    for category in range(1, num_channels):
        if category in labels:
            semantic_map_2d[:,:,category] = (model_output['masks'].squeeze(1)[labels == category].detach().cpu().max(dim=0)[0])[0]
    return semantic_map_2d[:,:, 1:].numpy()
