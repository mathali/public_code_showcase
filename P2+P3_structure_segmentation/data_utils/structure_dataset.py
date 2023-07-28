import joblib
import numpy as np
import torch
import random

from glob import glob
from torch.utils.data import Dataset


class IKEMDataset(Dataset):
    def __init__(self,
                 data_dir: str = 'F:/Halinkovic/IKEM/training_data/',
                 mode: str = 'binary',
                 nodst: bool = False,
                 colorspace: str = 'rgb'):
        self.data_dir = data_dir # + mode
        self.mode = mode
        self.nodst = nodst
        self.colorspace = colorspace
        if colorspace == 'lab':
            self.scalers = self.load_scalers()
        self.input_shape = (256, 256)

        self.patch_list = self.get_patch_list()
        self.mapping_dict = {
                            "Endocarium": 1,
                            "Blood vessels": 2,
                            "Inflammation": 3,
                            "Fatty tissue": 4,
                            "Immune cells": 3,
                            "Quilty": 3,
                            "Fibrotic tissue": 1,
                        }

    def __len__(self):
        return len(self.patch_list)

    def load_scalers(self):
        scalers = []

        if self.mode == 'binary':
            scalers.append(joblib.load('./scalers/multiclass/L_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multiclass/A_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multiclass/B_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multiclass/binary_dst_scaler.joblib'))
        elif self.mode == 'multiclass':
            scalers.append(joblib.load('./scalers/multiclass/L_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multiclass/A_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multiclass/B_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multiclass/multiclass_dst1_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multiclass/multiclass_dst2_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multiclass/multiclass_dst3_scaler.joblib'))

        return scalers

    def get_patch_list(self):
        random.seed(42)
        patch_list = glob(self.data_dir + '/*.npy')
        return patch_list


    def __getitem__(self, idx):
        patch = self.patch_list[idx]
        patch = np.load(patch)

        if self.colorspace == 'lab' and self.mode == 'binary':
            img = np.zeros((5, 256, 256), dtype=np.float32)
            img[0] = self.scalers[0].transform(patch[0].reshape(-1, 1)).reshape(256, 256)
            img[1] = self.scalers[1].transform(patch[1].reshape(-1, 1)).reshape(256, 256)
            img[2] = self.scalers[2].transform(patch[2].reshape(-1, 1)).reshape(256, 256)

            img[3, :, :] = patch[5, :, :]
            img[4] = self.scalers[3].transform(patch[6].reshape(-1, 1)).reshape(256, 256)

        elif self.colorspace == 'lab' and self.mode == 'multiclass':
            img = np.zeros((9, 256, 256))
            img[0] = self.scalers[0].transform(patch[0].reshape(-1, 1)).reshape(256, 256)
            img[1] = self.scalers[1].transform(patch[1].reshape(-1, 1)).reshape(256, 256)
            img[2] = self.scalers[2].transform(patch[2].reshape(-1, 1)).reshape(256, 256)

            img[3, :, :] = patch[5, :, :]
            img[4] = self.scalers[3].transform(patch[6].reshape(-1, 1)).reshape(256, 256)
            img[5, :, :] = patch[7, :, :]
            img[6] = self.scalers[4].transform(patch[8].reshape(-1, 1)).reshape(256, 256)
            img[7, :, :] = patch[9, :, :]
            img[8] = self.scalers[5].transform(patch[10].reshape(-1, 1)).reshape(256, 256)

        labels = patch[4, :, :]
        labels = np.where((labels == 5) | (labels == 6), 3, labels)
        labels = np.where(labels == 7, 1, labels)
        labels = np.where(labels == 2, 0, labels)
        labels = np.where(labels == 4, 2, labels)
        labels = np.where(labels == 3, 1, 0)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(labels, dtype=torch.uint8)

