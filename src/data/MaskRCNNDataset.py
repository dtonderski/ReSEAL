from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import os
import torch
from PIL import Image
import numpy as np
from src.model.perception.labeler import LabelGenerator
from torch import Tensor
from typing import List, Tuple, Dict
import quaternion
from typing import Optional
from pathlib import Path
from src.config import CfgNode
from src.data import filepath
import _pickle as cPickle
class MaskRCNNDataset(Dataset):

    @property
    def label_generator(self):
        return self._label_generator
    
    @label_generator.setter
    def label_generator(self, label_generator):
        self._label_generator = label_generator

    def __init__(self, data_paths_cfg: CfgNode, scene_split:str, epoch_number, transforms = MaskRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()):
        self._transforms = transforms
        self._data_paths_cfg = data_paths_cfg
        

        # might use data_path_cfg
        epoch_dir_path = filepath.get_trajectory_data_epoch_dir(data_paths_cfg, epoch_number)
        self._root = epoch_dir_path
        scene_ids = list(sorted([f.name for f in os.scandir(epoch_dir_path) if f.is_dir()]))

        imgs_paths = []
        label_dict_paths = []
        for scene in scene_ids:
            trajectory_output_dir = Path(data_paths_cfg.TRAJECTORIES_DIR) / f'epoch_{epoch_number}' / scene
            scene_rgb_path = trajectory_output_dir / 'RGB'
            imgs_paths += [scene_rgb_path/ path for path in list(sorted(os.listdir(scene_rgb_path)))]
            scene_label_dict_path = trajectory_output_dir / "LabelDicts"
            label_dict_paths += [scene_label_dict_path / path for path in list(sorted(os.listdir(scene_label_dict_path)))]
            
        self._img_paths = imgs_paths
        self._label_dict_paths = label_dict_paths


    def __getitem__(self, idx):
        img_path = self._img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self._transforms(img)

        label_dict_path = self._label_dict_paths[idx]
        with open(label_dict_path, 'rb') as fp:
            label_dict = cPickle.load(fp)

        return img, label_dict 

    def __len__(self):
        return len(self._img_paths)

def collate_fn(batch: List[Tuple[Tensor, Dict[str, Tensor]]]):
    return torch.stack([t for t, _ in batch]), [d for _, d in batch]