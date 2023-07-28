import glob
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset
from os.path import splitext

class NucleiDataset(Dataset):
    """
       Handles all the data needs of this nucleus classification project.
       Very simple dataset, only performs the following tasks: globs all prepared patches, stacks together input images
       and masks, passes them to the dataloader
    """
    def __init__(self, data_dir: str, mode: str = 'train'):
        self.data_dir = data_dir
        self.mode = mode

        self.__prepare_data__()
        self.input_shape = (15, 15)

    def __len__(self):
        return len(self.file_list)

    def __prepare_data__(self):
        self.file_list = glob.glob(self.data_dir + '/*.npy')
        self.file_list.sort()

    def set_mode(self, mode):
        self.mode = mode
        self.__prepare_data__()

    def preprocess(self, img):
        img = img.transpose((2, 1, 0))
        return img

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        img = np.load(self.file_list[idx])
        file_name = self.file_list[idx]
        c = file_name.split('=')[2].split('_')[0]
        parent = file_name.split('=')[1].split('_c')[0]
        y = file_name.split('=')[3].split('_')[0]
        x = file_name.split('=')[4].split('.')[0].split('_')[0]

        img = self.preprocess(img)

        return {
            'image': torch.as_tensor(np.copy(img), dtype=torch.float32).contiguous(),
            'class': int(c),
            'parent': parent,
            'y': int(y),
            'x': int(x)
        }
