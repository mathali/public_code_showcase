import numpy as np
import torch

from glob import glob
from torch.utils.data import Dataset


class IKEMCellDataset(Dataset):
    def __init__(self,
                 data_dir: str = 'F:/Halinkovic/IKEM/cell_training_data/',
                 mode: str = 'binary'):
        self.data_dir = data_dir
        self.mode = mode

        self.datasets = self.prepare_datasets()
        self.input_shape = (256, 256)

        self.patch_list = self.get_patch_list()
        self.mapping_dict = {
                                "Immune cells": 1,
                                "Muscle cells": 2,
                                "Other cells": 3,
                            }

    def __len__(self):
        return len(self.patch_list)

    def get_patch_list(self):
        return glob(self.data_dir + '/*.npy')

    def __getitem__(self, idx):
        patch = self.patch_list[idx]

        patch = np.load(patch)
        img = patch[:3, :, :] / 255
        labels = patch[4, :, :]

        if self.mode == 'binary':
            labels = np.where(labels != 0, 1, 0)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(labels, dtype=torch.uint8)
