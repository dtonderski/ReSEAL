from torchvision.io import read_image
from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import numpy as np

class MaskRCNNDataset(Dataset):

    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "RGB"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "MASKS"))))

    

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "RGB", self.imgs[idx])
        # Label_Generator
        mask_path = os.path.join(self.root, "MASKS", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = np.load(mask_path)
        #mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes =  []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, xmax, ymin, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None :
            img, target = self.transforms(img, target)
        return img, target #self.transform(image)

    def __len__(self):
        return len(self.imgs)
