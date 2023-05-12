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
    return semantic_map_2d[:,:, 1:]



import numpy as np
from tqdm import tqdm

from src.config import default_map_builder_cfg, default_sim_cfg, default_data_paths_cfg
from src.features.mapping import SemanticMap3DBuilder
from src.utils import category_mapping
from src.utils.misc import get_semantic_map, semantic_map_to_categorical_map
from src.visualisation.semantic_map_visualization import visualize_map
from src.utils.category_mapping import get_instance_index_to_reseal_name_dict
from src.visualisation.instance_map_visualization import get_instance_colormap
from src.features.perception import propagate_labels
from src.utils.geometric_transformations import coordinates_to_grid_indices
from scipy.ndimage import label
from src.visualisation import instance_map_visualization

from src.utils.reference.engine import train_one_epoch, evaluate
import src.utils.reference.transforms as T
import src.utils.reference.utils as utils
from src.model.MaskRCNN import build_maskrcnn, generating_semantic_masks
from torchvision.io import read_image

from src.model.perception.morphological_transformations import (
    semantic_map_to_categorical_label_map,
    fill_holes,
    remove_small_objects
)


# used in the tutorial for fine-tuning Maskrcnn, but might be obsolet for RESEAL
def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def train_perception_model(perception_model, scene_dir):
    cfg = default_maskrcnn_cfg()
    dataset = MaskRCNNDataset(scene_dir, transforms=MaskRCNN_ResNet50_FPN_Weights.DEFAULT.transforms())
    dataset_test = MaskRCNNDataset(scene_dir, transforms=MaskRCNN_ResNet50_FPN_Weights.DEFAULT.transforms())
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset, indices[-50:])

    data_loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=cfg.SHUFFLE, num_workers=cfg.NUM_WORKERS,
                        collate_fn=utils.collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=cfg.NUM_WORKERS,
                        collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = cfg.NUM_CLASSES
    model = build_maskrcnn(num_classes)
    model.to(device)
    print(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.LEARNING_RATE, momentum=cfg.OPTIM_MOMENTUM, weight_decay=cfg.OPTIM_WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=cfg.OPTIM_STEP_SIZE,
                                                    gamma=cfg.OPTIM_GAMMA)

    num_epochs = cfg.NUM_EPOCHS
    for epoch in range(num_epochs):
        train_one_epoch(perception_model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(perception_model, data_loader_test, device)
    return perception_model

def label_pipeline(model, trajectory):
    cfg = default_maskrcnn_cfg()
    data_paths_cfg = default_data_paths_cfg()
    TRAJECTORY = trajectory#"00006-HkseAnWCgqk"
    DEPTH_MAP_DIR = f"./data/interim/trajectories/train/{TRAJECTORY}/D"
    RGB_IMAGE_DIR = f"./data/interim/trajectories/train/{TRAJECTORY}/RGB"
    MASK_IMAGE_DIR = f"./data/interim/trajectories/train/{TRAJECTORY}/MASKS"
    POSITIONS_FILE = f"./data/interim/trajectories/train/{TRAJECTORY}/positions.npy"
    ROTATIONS_FILE = f"./data/interim/trajectories/train/{TRAJECTORY}/rotations.npy"
    SEMANTIC_MAP_DIR = f"./data/interim/trajectories/train/{TRAJECTORY}/Semantic"
    trajectory_name = TRAJECTORY.split("-")[1]
    SEMANTIC_INFO_FILE = f"./data/raw/train/scene_datasets/hm3d/train/{TRAJECTORY}/{trajectory_name}.semantic.txt"
    
    scene_to_reseal_mapping = category_mapping.get_scene_index_to_reseal_index_map_vectorized(SEMANTIC_INFO_FILE)
    
    sim_cfg = default_sim_cfg()
    map_builder_cfg = default_map_builder_cfg()
    map_builder_cfg.NUM_SEMANTIC_CLASSES = 6
    map_builder_cfg.RESOLUTION = 0.05
    map_builder_cfg.MAP_SIZE = [15, 1.5, 15]
    map_builder = SemanticMap3DBuilder(map_builder_cfg, sim_cfg)
        
    # model = build_maskrcnn(cfg.NUM_CLASSES)
    model.eval()

    rotations = np.load(ROTATIONS_FILE).view(dtype=np.quaternion)
    positions = np.load(POSITIONS_FILE)
    scene_index_to_category_index_map = category_mapping.get_scene_index_to_reseal_index_map_vectorized(SEMANTIC_INFO_FILE)

    map_builder.clear()
    for i in tqdm(range(400)):
        depth_map = np.load(f"{DEPTH_MAP_DIR}/{i}.npy")
        saved_semantics = np.load(f"{SEMANTIC_MAP_DIR}/{i}.npy")
        image = read_image(f"{RGB_IMAGE_DIR}/{i}.png")
        # map = generating_semantic_masks(model, image)
        map = get_semantic_map(saved_semantics, scene_index_to_category_index_map, 
                map_builder_cfg.NUM_SEMANTIC_CLASSES)
        pose = (positions[i], rotations[i])
        map_builder.update_point_cloud(map, depth_map, pose, fast=True)
    map_builder.concatenate_semantics()
    print("Concatenating semantics done")
    map_builder.update_kdtree()

    # point_cloud = np.asarray(map_builder.point_cloud.points)
    # point_cloud_semantic_labels = map_builder._point_cloud_semantic_labels
    for index in tqdm(range(0, 400)):
        map_at_index_sparse = map_builder.get_semantic_map_sparse((positions[index], 0), use_dicts=False)
        
        categorical_label_map = semantic_map_to_categorical_label_map(map_at_index_sparse, no_object_threshold=0.5)
        categorical_label_map = fill_holes(categorical_label_map, 10)
        categorical_label_map = remove_small_objects(categorical_label_map, 30)

        instance_map_instances, num_features = label(categorical_label_map[..., 1])
        instance_map = np.stack([categorical_label_map[..., 0], instance_map_instances], axis=-1)
        sensor_position = positions[index]
        sensor_rotation = rotations[index]

        min_position, _ = map_builder.get_semantic_map_bounds(positions[index])
        grid_index_of_min_position_relative_to_origin = coordinates_to_grid_indices(
            np.array(min_position), [0, 0, 0], map_builder._resolution
        )
        grid_index_of_origin = -grid_index_of_min_position_relative_to_origin

        instance_map_2d = propagate_labels(
            sensor_rotation, sensor_position, instance_map, grid_index_of_origin, map_builder_cfg, sim_cfg.SENSOR_CFG
        )
        np.save(MASK_IMAGE_DIR+"/"+str(index), instance_map_2d)
    return