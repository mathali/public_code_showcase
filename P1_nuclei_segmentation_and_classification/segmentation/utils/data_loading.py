import glob
import numpy as np
import torch

from os.path import splitext
from imgaug import augmenters as iaa
from PIL import Image
from torch.utils.data import Dataset

from .augs import (
    add_to_brightness,
    add_to_contrast,
    add_to_hue,
    add_to_saturation,
    gaussian_blur,
    median_blur,
)


class BasicDataset(Dataset):
    """
    Handles all the data needs of this nucleus segmentation project.
    Consists of two parts:
        1. Data loading - __prepare_data__ + __get_item__ : globs all prepared patches, stacks together input images
            and masks, passes them to the dataloader
        2. Preprocessing - __get_augmentation : image augmentations based on HoVer-Net - applies imgaug functions before
            passing the image data to the data loader. Can be turned off in config.json

    """
    def __init__(self, data_dir: str, scale: float = 1.0,augment: bool = True, mode: str = 'train'):
        self.data_dir = data_dir
        self.mode = mode

        self.__prepare_data__()
        self.input_shape = (256, 256)

        self.scale = scale

        self.augmentor = None
        self.shape_augs = None
        self.input_augs = None
        self.augment = augment
        self.__augmentation_setup__()

    def __len__(self):
        return len(self.file_list)

    def __augmentation_setup__(self):
        if self.augment:
            self.augmentor = self.__get_augmentation(self.mode, 0)
            self.shape_augs = iaa.Sequential(self.augmentor[0])
            self.input_augs = iaa.Sequential(self.augmentor[1])

    def __prepare_data__(self):
        self.file_list = glob.glob(self.data_dir + f'/{self.mode}' + '/*.npy')
        self.file_list.sort()

    def set_mode(self, mode):
        self.mode = mode
        self.__prepare_data__()
        self.__augmentation_setup__()

    def preprocess(self, img, mask):
        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            img = shape_augs.augment_image(img)
            mask = shape_augs.augment_image(mask)

        img = img.transpose((2, 0, 1))
        mask = np.where(mask > 0, 1, mask)

        return img, mask

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

        data_mat = np.load(self.file_list[idx])

        img = (data_mat[..., :3]).astype(np.float32)
        mask = (data_mat[..., 4]).astype(np.uint8)

        img, mask = self.preprocess(img, mask)

        classes = (data_mat[..., 4]).astype(np.uint8)
        return {
            'image': torch.as_tensor(np.copy(img), dtype=torch.float32).contiguous(),
            'mask': torch.as_tensor(np.copy(mask), dtype=torch.uint8).contiguous(),
            'file': self.file_list[idx],
            'classes': classes
        }

    def __get_augmentation(self, mode, rng=0):
        if self.mode == "train":
            shape_augs = [
                iaa.Affine(
                    # scale images to 80-120% of their size, individually per axis
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # translate by -A to +A percent (per axis)
                    translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                    shear=(-5, 5),  # shear by -5 to +5 degrees
                    rotate=(-179, 179),  # rotate by -179 to +179 degrees
                    order=0,  # use nearest neighbour
                    backend="cv2",  # opencv for fast processing
                    seed=rng,
                ),
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                ),
                iaa.Fliplr(0.5, seed=rng),
                iaa.Flipud(0.5, seed=rng),
            ]

            input_augs = [
                iaa.OneOf(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: gaussian_blur(*args, max_ksize=3),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: median_blur(*args, max_ksize=3),
                        ),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),
                    ]
                ),
                iaa.Sequential(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_hue(*args, range=(-8, 8)),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_saturation(
                                *args, range=(-0.2, 0.2)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_brightness(
                                *args, range=(-26, 26)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_contrast(
                                *args, range=(0.75, 1.25)
                            ),
                        ),
                    ],
                    random_order=True,
                ),
            ]
        elif self.mode == "valid" or self.mode == "test":
            shape_augs = [
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                )
            ]
            input_augs = []

        return shape_augs, input_augs