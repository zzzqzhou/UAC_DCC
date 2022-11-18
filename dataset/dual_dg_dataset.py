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
    def __init__(self, mode, ids=None, size=None, dataset=None):
        self.mode = mode
        self.ids = ids
        self.size = size
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

        img_s1, img_s2 = deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)

        img_s1, mask = normalize(img_s1, mask)
        img_s2 = normalize(img_s2)
        
        return img_s1, img_s2, mask
    
    def __len__(self):
        return len(self.ids)
