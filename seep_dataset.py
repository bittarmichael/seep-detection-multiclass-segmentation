import os
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset
from utils import ImageHelperFunctions, ImageTransformFunctions
from torchvision import transforms

class SeepImageDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.helper = ImageHelperFunctions()
        self.transform = ImageTransformFunctions()
        self.list_img_paths = self.helper.list_files_in_dir(img_dir, ".tif")
        self.list_mask_paths = self.helper.list_files_in_dir(mask_dir, ".tif")
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(0.11172756, 0.33363158)
        
    def __len__(self):
        return len(self.list_img_paths)
    
    def __getitem__(self, index):
        image = self.helper.read_image(self.list_img_paths[index])
        mask = self.helper.read_mask(self.list_mask_paths[index])
        
        image_t, mask = self.transform.augment(image, mask)
        
        image = self.to_tensor(image_t)
        image = image / 65535
        image = self.normalize(image)
         
        mask = np.array(mask)
        mask = torch.from_numpy(mask)
        mask = mask.type(torch.LongTensor)
        
        return image, mask