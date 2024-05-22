#!/usr/bin/env python3

from . import images
import numpy as np
import torch
from torchvision import transforms

class SpecialSeqPairs(torch.utils.data.Dataset):
    def __init__(self, npz_path, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(256,256))])):
        super().__init__()
        data = np.load(npz_path)
        self.image_dataset = transform(np.moveaxis(data['images'], 3, 1))
        self.end_indices = data['motion_start'].astype(int) - 1
        self.end_indices[0] = len(self.image_dataset) - 1

    def __getitem__(self, index):
        if index not in self.end_indices:
            q1 = self.image_dataset[index]
            q2 = self.image_dataset[index + 1]
        else:
            q1 = q2 = self.image_dataset[index]
        return ((q1, q2), (q1, q2))

    def __len__(self):
        return len(self.image_dataset) - 1

def build(props):
    return SpecialSeqPairs(props['file'])
