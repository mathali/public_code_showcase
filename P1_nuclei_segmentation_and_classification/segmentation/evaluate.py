import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from utils.dice_score import multiclass_dice_coeff, dice_coeff, per_class_dice_coeff, dice_loss
from utils.distance_loss import dst_loss


"""
Function that handles validation for every epoch during training.
Once again - supports both 1 and 2 channel binary segmentation.
"""
def evaluate(net, dataloader, device, experiment, optimizer, global_step, epoch, histograms, criterion):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    multi_dice_score = [0] * net.n_classes

    total_loss = 0
    #### VALIDATION START ####
    # Iterates through the validation loader and calculates metrics
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.uint8)

        with torch.no_grad():
            mask_pred = net(image)

            if net.n_classes == 1:
                mask_true = mask_true.float()
            else:
                loss = criterion(mask_pred, mask_true.long()) \
                       + dice_loss(F.softmax(mask_pred, dim=1).float(),
                                   F.one_hot(mask_true.long(), net.n_classes).permute(0, 3, 1, 2).float(),
                                   multiclass=True) \
                       + dst_loss(F.softmax(mask_pred, dim=1).float(),
                                  F.one_hot(mask_true.long(), net.n_classes).permute(0, 3, 1, 2).float(),
                                  net.n_classes)
                mask_true = F.one_hot(mask_true.long(), net.n_classes).permute(0, 3, 1, 2).float()

            if net.n_classes == 1:
                mask_pred = torch.squeeze(mask_pred)
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                loss = criterion(mask_pred, mask_true.float()) \
                       + dice_loss(F.sigmoid(mask_pred).float(),
                                   mask_true.float(),
                                   multiclass=False) \
                       + dst_loss(F.sigmoid(mask_pred).float(),
                                  mask_true.float(), net.n_classes)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()

                dice_score += multiclass_dice_coeff(mask_pred[:, :, ...], mask_true[:, :, ...],
                                                    reduce_batch_first=False)

                tmp_per_class = per_class_dice_coeff(mask_pred[:, :, ...], mask_true[:, :, ...],
                                                    reduce_batch_first=False)


                for ind in range(len(tmp_per_class)):
                    multi_dice_score[ind] += tmp_per_class[ind]

            total_loss += loss.item()

    net.train()

    #### VALIDATION END ####

    #### LOGGING AND OUTPUT
    if net.n_classes == 1:
        if experiment is not None:
            experiment.log({
                'learning rate': optimizer.param_groups[0]['lr'],
                'validation Dice': dice_score / num_val_batches,
                'validation loss': total_loss / num_val_batches,
                'images': wandb.Image(image[0].cpu()),
                'masks': {
                    'true': wandb.Image(mask_true[0].float().cpu()),
                    'pred': wandb.Image((F.sigmoid(mask_pred)[0] > 0.5).float().cpu()),
                },
                'step': global_step,
                'epoch': epoch,
                **histograms
            })
        return dice_score / num_val_batches
    else:
        class_dice = {}
        for c in range(len([x/num_val_batches for x in multi_dice_score])):
            class_dice[f'class_{c}_dice'] = [x/num_val_batches for x in multi_dice_score][c]
        if experiment is not None:
            experiment.log({
                'learning rate': optimizer.param_groups[0]['lr'],
                'validation Dice': dice_score / num_val_batches,
                'validation loss': total_loss / num_val_batches,
                'per class Dice': class_dice,
                'images': wandb.Image(image[0].cpu()),
                'masks': {
                    'true': wandb.Image(mask_true[0].float().cpu()),
                    'pred': wandb.Image(torch.softmax(mask_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                },
                'step': global_step,
                'epoch': epoch,
                **histograms
            })
        return dice_score / num_val_batches, [x/num_val_batches for x in multi_dice_score]
