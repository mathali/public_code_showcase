from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
from utils.distance_loss import dst_loss
from utils.store_predictions import store_predictions
from evaluate import evaluate


def train_net(net,
              device,
              epochs: int = 50,
              batch_size: int = 32,
              learning_rate: float = 1e-4,
              save_checkpoint: bool = True,
              img_scale: float = 1.,
              amp: bool = False,
              data_dir: str = '',
              dir_checkpoint: str = 'outputs',
              experiment = None,
              normalizer: str = '',
              augment: bool = True):
    """ Primary training script
    Performs validation through "evaluate"
    Stores model at every checkpoint at which the model's validation performance improves
    Supports training for both 1 channel and 2 channel binary model output.

    Parameters are controlled using config.json
    """

    ### TRAINING SETUP START #####
    train_set = BasicDataset(data_dir, img_scale,  augment=augment, mode='train')
    val_set = BasicDataset(data_dir, img_scale, augment=augment, mode='valid')
    val_set.set_mode('valid')

    n_train = len(train_set)

    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)


    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-8)

    # Optional scheduler
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    if net.n_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    global_step = 0
    prev_best = 0

    ### TRAINING SETUP END #####

    # Main training loop
    for epoch in range(epochs):
        net.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            #### TRAINING START ####
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.uint8)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)

                    if net.n_classes == 1:
                        masks_pred = torch.squeeze(masks_pred)
                        loss = criterion(masks_pred, true_masks.float()) \
                               + dice_loss(F.sigmoid(masks_pred).float(),
                                           true_masks.float(),
                                           multiclass=False) \
                               + dst_loss(F.sigmoid(masks_pred).float(),
                                          true_masks.float(), net.n_classes)

                    else:
                        loss = criterion(masks_pred, true_masks.long()) \
                               + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                           F.one_hot(true_masks.long(), net.n_classes).permute(0, 3, 1, 2).float(),
                                           multiclass=True) \
                               + dst_loss(F.softmax(masks_pred, dim=1).float(),
                                          F.one_hot(true_masks.long(), net.n_classes).permute(0, 3, 1, 2).float(),
                                          net.n_classes)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1

                if experiment is not None:
                    experiment.log({
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                #### TRAINING END ####

            #### VALIDATION START ####
            histograms = {}
            for tag, value in net.named_parameters():
                tag = tag.replace('/', '.')
                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

            if net.n_classes == 1:
                val_score = evaluate(net, val_loader, device, experiment, optimizer, global_step, epoch, histograms, criterion)

            else:
                val_score, per_class = evaluate(net, val_loader, device, experiment, optimizer, global_step, epoch, histograms, criterion)

            # scheduler.step(val_score)

        if save_checkpoint and val_score > prev_best:
            prev_best = val_score
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            best_model = net
            torch.save(net.state_dict(), f'{dir_checkpoint}/checkpoint_epoch{epoch+1}.pth')

        #### VALIDATION END ####

    # Optional section that stores the final version of the model's validation predictions
    # store_predictions(best_model, val_loader, device, dir_checkpoint)
