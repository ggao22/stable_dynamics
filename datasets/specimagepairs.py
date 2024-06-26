#!/usr/bin/env python3

import logging

from . import images
import numpy as np
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)

class SpecialSeqPairs(torch.utils.data.Dataset):
    def __init__(self, npz_path, 
                    transform=transforms.Compose([transforms.ToPILImage(), 
                                                    transforms.ToTensor()])):
        super().__init__()
        data = np.load(npz_path, allow_pickle=True)
        self.image_dataset = data['data'][()]['img'][:,:,:,[2,1,0]]
        self.end_indices = (data['meta'][()]['episode_ends'] - 1).astype(int)
        # self.end_indices[0] = len(self.image_dataset) - 1
        self.transform = transform

    def __getitem__(self, index):
        if index not in self.end_indices:
            q1 = self.transform(self.image_dataset[index])
            q2 = self.transform(self.image_dataset[index + 1])
        else:
            q1 = self.transform(self.image_dataset[index - 1])
            q2 = self.transform(self.image_dataset[index])
        return ((q1, q2), (q1, q2))

    def __len__(self):
        return len(self.image_dataset)

def build(props):
    return SpecialSeqPairs(props['file'])
