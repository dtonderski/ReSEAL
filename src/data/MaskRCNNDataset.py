from torchvision.io import read_image
from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import numpy as np
from src.model.perception.labeler import LabelGenerator
import cv2
class MaskRCNNDataset(Dataset):

    @property
    def label_generator(self):
        return self._label_generator
    
    @label_generator.setter
    def label_generator(self, label_generator):
        self._label_generator = label_generator

    def __init__(self, root, transforms=None, label_generator=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "RGB"))))
        POSITIONS_FILE = f"{root}/positions.npy"
        ROTATIONS_FILE = f"{root}/rotations.npy"

        self.rotations = np.load(ROTATIONS_FILE).view(dtype=np.quaternion)
        self.positions = np.load(POSITIONS_FILE)   
        #self.masks = list(sorted(os.listdir(os.path.join(root, "MASKS"))))
        self._label_generator = label_generator    

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "RGB", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)
        pose = (self.positions[idx], self.rotations[idx])
        
        if self._label_generator is None:
            print("No LabelGenerator defined for the MaskRCNN Dataset")

        label_dict = self._label_generator.get_label_dict(pose)
        return img, label_dict #self.transform(image)

    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    return tuple(zip(*batch))