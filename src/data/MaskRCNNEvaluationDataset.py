from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import os
import torch
from PIL import Image
import numpy as np
from torch import Tensor
from typing import List, Tuple, Dict
from nptyping import Bool, Float, NDArray, Shape, Int
import quaternion
from typing import Optional
from pathlib import Path
from src.config import CfgNode
from src.data import filepath
import _pickle as cPickle

class MaskRCNNEvaluationDataset(Dataset):

    def __init__(self, data_paths_cfg: CfgNode, scene_split:str, epoch: int, 
                 transforms = MaskRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()):
        self._transforms = transforms
        self._data_paths_cfg = data_paths_cfg
        

        # might use data_path_cfg
        epoch_dir_path = filepath.get_trajectory_data_epoch_dir(data_paths_cfg, epoch)
        self._root = epoch_dir_path
        scene_ids = list(sorted([f.name for f in os.scandir(epoch_dir_path) if f.is_dir()]))

        imgs_paths = []
        semantic_paths = []
        scene_info_paths = []
        for scene in scene_ids:
            SEMANTIC_INFO_PATH = (Path('data') / 'raw' / 'val' / 'scene_datasets' / 'hm3d' / 'val' 
                      / scene / f'{scene.split("-")[1]}.semantic.txt')
            trajectory_output_dir = Path(data_paths_cfg.TRAJECTORIES_DIR) / f'epoch_{epoch}' / scene
            scene_rgb_path = trajectory_output_dir / 'RGB'
            imgs_paths += [scene_rgb_path/ path for path in list(sorted(os.listdir(scene_rgb_path)))]
            scene_semantic_path = trajectory_output_dir / "Semantic"
            semantic_paths += [scene_semantic_path / path for path in list(sorted(os.listdir(scene_semantic_path)))]
            scene_info_paths += [SEMANTIC_INFO_PATH for path in list(sorted(os.listdir(scene_semantic_path)))]
            
        self._img_paths = imgs_paths
        self._semantic_paths = semantic_paths
        self._scene_info_paths = scene_info_paths

    def __getitem__(self, idx):
        img_path = self._img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self._transforms(img)

        semantic_path = self._semantic_paths[idx]
        semantic = np.load(semantic_path)
        scene_info_path = self._scene_info_paths[idx]
        return img, semantic, scene_info_path

    def __len__(self):
        return len(self._img_paths)

def eval_collate_fn(batch):
    return torch.stack([t for t, _, _ in batch]), [d for _, d, _ in batch], [s for _, _, s in batch]