import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np


class MiniBatchCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    Loss function described in Section 6.3.2

    Balances cross entropy so that each class has equal contributions to the final loss in each batch.
    """
    log_softmax = nn.LogSoftmax()
    device = torch.device("cuda:"+str(0)) if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = F.cross_entropy(input, target, reduction='none')
        c_b, counts = torch.unique(target, return_counts=True)

        c_ij = torch.tensor(np.array([0, 0, 0, 0])).to(self.device)
        for var, val in zip(c_b.cpu().numpy(), counts.cpu().numpy()):
            c_ij[var] = val
        c_b = torch.tensor(data=c_b.shape[0], dtype=torch.uint8).to(self.device)
        l_b = torch.sum(c_ij)

        c_ij = c_ij.cpu().numpy()
        c_b = c_b.cpu().numpy()
        l_b = l_b.cpu().numpy()

        calc_weights = lambda x: l_b / (c_b * c_ij[x])

        target = target.cpu().numpy()
        weights = calc_weights(target)
        weights = np.where(target == 0, weights, 1)
        weights = torch.tensor(data=weights).to(self.device)

        weighted_loss = torch.mean(loss*weights)

        return weighted_loss


##############################################################################################
# Standard Dice calculations used throught the projects

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(0, input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def per_class_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Dice coefficient for each class
    assert input.size() == target.size()
    dice = [0] * input.shape[1]
    for channel in range(input.shape[1]):
        dice[channel] = dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
