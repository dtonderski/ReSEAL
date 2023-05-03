from torchvision.io import read_image
from torch.utils.data import Dataset
import os

class MaskRCNNDataset(Dataset):

    def __init__(self, img_dir, size, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        path = self.img_dir + '/'+str(idx)+'.png'
        image = read_image(path)
        return image #self.transform(image)
