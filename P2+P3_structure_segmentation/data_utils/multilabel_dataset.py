import joblib
import numpy as np
import torch
import random

from glob import glob
from torch.utils.data import Dataset


class IKEMDataset(Dataset):
    def __init__(self,
                 data_dir: str = 'C:/IKEM_training_datasets/multilabeling_v1/{mode}/{xy}/',
                 mode: str = 'binary',
                 nodst: bool = False,
                 colorspace: str = 'rgb'):
        self.data_dir = data_dir
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

        if self.mode == 'multiclass':
            scalers.append(joblib.load('./scalers/multilabel/L_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multilabel/A_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multilabel/B_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multilabel/multiclass_dst1_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multilabel/multiclass_dst2_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multilabel/multiclass_dst3_scaler.joblib'))
        elif self.mode == 'multiclass_pyramid':
            scalers.append(joblib.load('./scalers/multilabel/L_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multilabel/A_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multilabel/B_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multilabel/multiclass_dst1_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multilabel/multiclass_dst2_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multilabel/multiclass_dst3_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multilabel/L_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multilabel/A_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multilabel/B_scaler.joblib'))

        return scalers

    def get_patch_list(self):
        random.seed(42)
        patch_list = glob(self.data_dir.format(xy='data') + '/*.npy')
        return patch_list

    def __getitem__(self, idx):
        patch = self.patch_list[idx]
        labels = np.load(self.data_dir.format(mode='valid', xy='labels') + '/' + patch.split("\\")[-1])
        patch = np.load(patch)

        if self.colorspace == 'lab' and self.mode == 'binary':
            img = np.zeros((5, 256, 256), dtype=np.float32)
            img[0] = self.scalers[0].transform(patch[0].reshape(-1, 1)).reshape(256, 256)
            img[1] = self.scalers[1].transform(patch[1].reshape(-1, 1)).reshape(256, 256)
            img[2] = self.scalers[2].transform(patch[2].reshape(-1, 1)).reshape(256, 256)

            img[3, :, :] = patch[5, :, :]
            img[4] = self.scalers[3].transform(patch[6].reshape(-1, 1)).reshape(256, 256)

        elif self.colorspace == 'lab' and self.mode == 'multiclass':
            img = np.zeros((6, 256, 256))
            img[0] = self.scalers[0].transform(patch[0].reshape(-1, 1)).reshape(256, 256)
            img[1] = self.scalers[1].transform(patch[1].reshape(-1, 1)).reshape(256, 256)
            img[2] = self.scalers[2].transform(patch[2].reshape(-1, 1)).reshape(256, 256)

            img[3] = self.scalers[3].transform(patch[5].reshape(-1, 1)).reshape(256, 256)
            img[4] = self.scalers[4].transform(patch[7].reshape(-1, 1)).reshape(256, 256)
            img[5] = self.scalers[5].transform(patch[9].reshape(-1, 1)).reshape(256, 256)

        elif self.colorspace == 'lab' and self.mode == 'multiclass_pyramid':
            img = np.zeros((9, 256, 256))
            img[0] = self.scalers[0].transform(patch[0].reshape(-1, 1)).reshape(256, 256)
            img[1] = self.scalers[1].transform(patch[1].reshape(-1, 1)).reshape(256, 256)
            img[2] = self.scalers[2].transform(patch[2].reshape(-1, 1)).reshape(256, 256)

            img[3] = self.scalers[3].transform(patch[4].reshape(-1, 1)).reshape(256, 256)
            img[4] = self.scalers[4].transform(patch[5].reshape(-1, 1)).reshape(256, 256)
            img[5] = self.scalers[5].transform(patch[6].reshape(-1, 1)).reshape(256, 256)

            img[6] = self.scalers[6].transform(patch[7].reshape(-1, 1)).reshape(256, 256)
            img[7] = self.scalers[7].transform(patch[8].reshape(-1, 1)).reshape(256, 256)
            img[8] = self.scalers[8].transform(patch[9].reshape(-1, 1)).reshape(256, 256)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(labels, dtype=torch.uint8)
