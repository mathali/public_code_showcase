import torch
from torch import Tensor
import cv2 as cv
import numpy as np


def dst_loss(input: Tensor, target: Tensor, n_classes):
    # Custom distance loss function, described by Algorithm 1 in the thesis.
    # Penalizes false positive predictions based on their distance from the nearest true positive

    target_np = target.cpu().detach().numpy().astype(np.uint8)
    target_np = np.where(target_np > 0, 1, target_np)

    input_np = input.cpu().detach().numpy().astype(np.uint8)

    xor = np.logical_xor(input_np, target_np)
    px = np.logical_and(input_np, xor)

    target_np = ~target_np.astype(bool)

    total_loss = 0
    for c in range(n_classes):
        for mask, true in zip(px, target_np):
            if n_classes > 1:
                dst = cv.distanceTransform(true.astype(np.uint8)[c, :, :], cv.DIST_L2, 3)
            else:
                dst = cv.distanceTransform(true.astype(np.uint8), cv.DIST_L2, 3)
            maxval = np.amax(dst)
            dst = np.where(dst > maxval/3, maxval, dst)
            dst_loss_mask = dst * mask
            total_loss += dst_loss_mask.sum() / dst_loss_mask.size

    return total_loss / target_np.shape[0]