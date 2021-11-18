import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetClass(Dataset):
    def __init__(self, root=''):
        super(DatasetClass, self).__init__()
        self.img_files = glob.glob(os.path.join(root, 'image', '*.png'))
        self.mask_files = []
        for img_path in self.img_files:
            basename = os.path.basename(img_path)
            self.mask_files.append(os.path.join(root, 'mask', basename[:-4]+'_mask.png'))

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        data = np.expand_dims(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), 0)
        label = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        return torch.from_numpy(data).float(), torch.from_numpy(label).long()

    def __len__(self):
        return len(self.img_files)
