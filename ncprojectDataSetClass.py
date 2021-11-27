import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from numpy import random

class DatasetClass(Dataset):
    def __init__(self, root='',transform = False):
        super(DatasetClass, self).__init__()
        self.img_files = glob.glob(os.path.join(root, 'image', '*.png'))
        self.mask_files = []
        self.transformations = transform
        for img_path in self.img_files:
            basename = os.path.basename(img_path)
            self.mask_files.append(os.path.join(root, 'mask', basename[:-4]+'_mask.png'))

    def AugmentData(self,data,label):
        prob = random.rand(1, 3)[0]

        #Rotate image with any angle between [-5,5] if the probablity is greater than 30%
        if prob[0] >= 0.3:
            angle = random.randint(-5,5,1)
            (h, w) = data.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D((cX, cY), float(angle), 1.0)
            data = cv2.warpAffine(data, M, (w, h))
            label = cv2.warpAffine(label, M, (w, h))

        # Flip image vertically if probability is greater than 30
        if prob[1] >= 0.3:
            data = cv2.flip(data, 0)
            label = cv2.flip(label, 0)

        #Flip image Horizontally if probability is greater than 30
        if prob[2] >=0.3:
            data = cv2.flip(data, 1)
            label = cv2.flip(label, 1)

        return data,label


    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]

        data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        label = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        if self.transformations:
            data,label = self.AugmentData(data,label) 

        data = np.expand_dims(data, 0)
        return torch.from_numpy(data).float()/255, torch.from_numpy(label).long()   # Normalize pixels to lie between [0, 1]

    def __len__(self):
        return len(self.img_files)
