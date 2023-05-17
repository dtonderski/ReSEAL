from torchvision.io import read_image
from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import numpy as np
from src.model.perception.labeler import LabelGenerator
import cv2
class MaskRCNNDataset(Dataset):

    def __init__(self, root, transforms=None, label_generator=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "RGB"))))
        POSITIONS_FILE = f"{root}/positions.npy"
        ROTATIONS_FILE = f"{root}/rotations.npy"

        self.rotations = np.load(ROTATIONS_FILE).view(dtype=np.quaternion)
        self.positions = np.load(POSITIONS_FILE)   
        #self.masks = list(sorted(os.listdir(os.path.join(root, "MASKS"))))
        self.label_generator = label_generator

    

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "RGB", self.imgs[idx])
        #img = Image.open(img_path).convert("RGB")

        rgb_image = cv2.imread(img_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        img = rgb_image / 255
        #img = self.transforms(img)
        pose = (self.positions[idx], self.rotations[idx])
        #instance_map_2d = self.label_generator.get_instance_map_2d(pose)
        
        if self.label_generator is None:
            print("No LabelGenerator defined for the MaskRCNN Dataset")

        label_dict = self.label_generator.get_label_dict(pose)
        print(label_dict['boxes'].shape)
        image_id = torch.tensor([idx])
        return img, label_dict #self.transform(image)

    def __len__(self):
        return len(self.imgs)
