import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
import numpy as np
import os
from utils.dice_score import multiclass_dice_coeff, dice_coeff, per_class_dice_coeff

# Basically the same concept as evaluate.py, except this also stores the output
def store_predictions(net, dataloader, device, dir_checkpoint):
    net.eval()
    num_val_batches = len(dataloader)
    out_path = f'{dir_checkpoint}/stored_output/'
    os.mkdir(f'{dir_checkpoint}/stored_output')
    dice_score = 0
    multi_dice_score = [0] * net.n_classes

    batch_count = 0
    for batch in tqdm(dataloader, total=num_val_batches, desc='Saving output', unit='batch', leave=False):
        #### DATA PREDICTION START ####
        image, mask_true = batch['image'], batch['mask']
        files = batch['file']
        classes = batch['classes']

        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.uint8)
        if net.n_classes == 1:
            mask_true = mask_true.float()
        else:
            mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            mask_pred = net(image)

            if net.n_classes == 1:
                mask_pred = torch.squeeze(mask_pred)
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(mask_pred[:, :, ...], mask_true[:, :, ...],
                                                    reduce_batch_first=False)

                tmp_per_class = per_class_dice_coeff(mask_pred[:, :, ...], mask_true[:, :, ...],
                                                    reduce_batch_first=False)

                for ind in range(len(tmp_per_class)):
                    multi_dice_score[ind] += tmp_per_class[ind]

        #### DATA PREDICTION END ####

        #### DATA STORAGE START ####
        img_count = 0
        for img, true, pred, file, c in zip(image.cpu().detach().numpy(),
                                         mask_true.cpu().detach().numpy(),
                                         mask_pred.cpu().detach().numpy(),
                                         files, classes):
            true = true.reshape((1, 256, 256))
            pred = pred.reshape((1, 256, 256))
            c = c.reshape((1, 256, 256))

            out = np.concatenate((img, true, pred, c))
            out = np.transpose(out, (1, 2, 0))

            file = file.split('\\')[-1]
            np.save(out_path+f"batch={batch_count}_img={file}.npy", out)
            img_count += 1
        batch_count += 1
        #### DATA STORAGE END ####

    print('Test Dice score: {}'.format(dice_score / num_val_batches))
