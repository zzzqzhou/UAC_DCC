from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

category_list = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
                 "traffic sign", "vegetation", "terrain", "sky", "person", "rider",
                 "car", "truck", "bus", "train", "motorcycle", "bicycle"]

class DG_Dataset(Dataset):
    def __init__(self, mode, ids=None, size=None, is_sda=False, dataset=None):
        self.mode = mode
        self.ids = ids
        self.size = size
        self.is_sda = is_sda
        self.dataset = dataset
    
    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(id.split(' ')[0]).convert('RGB')
        mask = Image.fromarray(np.array(Image.open(id.split(' ')[1])))

        if self.mode == 'val':
            if self.dataset == 'gtav':
                img, mask = resize_val(img, mask, (1914, 1052))
            img, mask = normalize(img, mask)
            return img, mask, id
        
        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.is_sda:
            if random.random() < 0.8:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.2)(img)
            img = blur(img, p=0.5)
            img, mask = normalize(img, mask)

            return img, mask
        
        img, mask = normalize(img, mask)
        return img, mask
    
    def __len__(self):
        return len(self.ids)
