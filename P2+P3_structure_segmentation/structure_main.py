import torch
import torch.nn.functional as F
import numpy as np
import wandb

from tqdm import tqdm
from data_utils.structure_dataset import IKEMDataset
from torch.utils.data import DataLoader
from models.unet_model import UNet
from utils.balanced_losses import MiniBatchCrossEntropyLoss, multiclass_dice_coeff, per_class_dice_coeff, dice_loss
from configuration import configuration


def train_model():
    """ Primary training script
    Responsible for every part of the training process - training, validating, and testing
    Stores model at every checkpoint at which the model's validation performance improves

    Parameters are controlled using structure_config.json
    """

    ### TRAINING SETUP START #####
    config = configuration('./configs/structure_config.json')
    batch_size = config.batch_size
    lr = config.lr
    epochs = config.epochs
    log = config.log
    n_classes = config.n_classes
    l2_lambda = config.l2_lambda
    mode = config.mode
    norm = config.norm
    nodst = config.nodst
    colorspace = config.colorspace

    data_dir = f"./train_data/multiclass/{mode}"

    name = f"{mode.capitalize()}_cells_{colorspace.upper()}_{norm}norm_scaled_DCL_slow"
    train_set = IKEMDataset(data_dir=data_dir + '/train', mode=mode, nodst=nodst,
                          colorspace=colorspace)
    val_set = IKEMDataset(data_dir=data_dir + '/valid', mode=mode, nodst=nodst,
                            colorspace=colorspace)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=24,
                              pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=24,
                            pin_memory=True, prefetch_factor=2)

    device = torch.device("cuda:"+str(0)) if torch.cuda.is_available() else torch.device("cpu")

    if mode == 'binary':
        model = UNet(n_channels=5, n_classes=n_classes, norm=norm)
    elif mode == 'multiclass' and not nodst:
        model = UNet(n_channels=9, n_classes=n_classes, norm=norm)
    elif mode == 'multiclass' and nodst:
        model = UNet(n_channels=6, n_classes=n_classes, norm=norm)

    optimizer = torch.optim.Adam([p for p in model.parameters()], lr=lr, weight_decay=l2_lambda)
    criterion = MiniBatchCrossEntropyLoss()
    model = model.to(device)
    prev_val_loss = np.inf
    early_stopping = 0

    print(f'Total number of trainable parameters: {sum(p.numel() for p in model.parameters())}')

    experiment = None
    if log:
        wandb.login(key="YOUR-KEY-HERE")
        experiment = wandb.init(project='IKEM-Structure-Segmentation', resume='allow', name=name, group='lab',
                                config={
                                    "learning_rate": lr,
                                    "batch_size": batch_size,
                                    "epochs": epochs,
                                    "l2_lambda": l2_lambda,
                                    "n_classes": n_classes,
                                    "loss_func": "DiceLoss+MiniBatchCrossEntropy",
                                    "norm_layers": norm,
                                    "colorspace": colorspace,
                                    "scaling": "standard",
                                    "fat": True,
                                    "separate_validation": True
                                    }
                                )

    ### TRAINING SETUP END #####

    ### TRAINING START #####
    for e in range(1, epochs+1):
        model.train()
        train_loss_total, train_accuracy_total = 0, 0
        train_dice_score = 0
        train_multi_dice_score = [0] * n_classes

        with tqdm(total=len(train_loader), desc='Batch') as pbar:
            for data, labels in train_loader:
                data = data.to(device)
                labels = labels.to(device)

                pred = model(data)
                loss = criterion(pred.view(labels.size(dim=0), n_classes, 256, 256), labels.long()) \
                               + dice_loss(F.softmax(pred, dim=1).float(),
                                           F.one_hot(labels.long(), n_classes).permute(0, 3, 1, 2).float(),
                                           multiclass=True)

                mask_true = F.one_hot(labels.long(), n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(F.softmax(pred, dim=1).argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
                train_dice_score += multiclass_dice_coeff(mask_pred[:, :, ...], mask_true[:, :, ...],
                                                          reduce_batch_first=False)

                tmp_per_class = per_class_dice_coeff(mask_pred[:, :, ...], mask_true[:, :, ...],
                                                     reduce_batch_first=False)

                for ind in range(len(tmp_per_class)):
                    train_multi_dice_score[ind] += tmp_per_class[ind]

                pred = torch.argmax(F.softmax(pred.view(labels.size(dim=0), n_classes, 256, 256), dim=1).float(), dim=1)
                pred = pred.long().cpu().numpy().astype(np.uint8)

                weight = np.where(labels.cpu().numpy() == 255, 0, 1).astype(bool)
                accuracy = labels.cpu().numpy() == pred
                accuracy = accuracy[weight]
                accuracy = accuracy.sum() / accuracy.size

                train_loss_total += loss.item()
                train_accuracy_total += accuracy

                optimizer.zero_grad()

                if experiment is not None:
                    experiment.log({
                        'train loss': loss.item(),
                        'train accuracy': accuracy,
                    })

                loss.backward()
                optimizer.step()

                pbar.update(1)

        histograms = {}
        for tag, value in model.named_parameters():
            tag = tag.replace('/', '.')
            if value.grad is not None and experiment is not None:
                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

        ### TRAINING END #####

        ### VALIDATION START #####
        model.eval()
        val_loss_total, val_accuracy_total = 0, 0,
        val_dice_score = 0
        val_multi_dice_score = [0] * n_classes

        with tqdm(total=len(val_loader), desc='Batch') as pbar:
            for data, labels in val_loader:
                data = data.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    pred = model(data)
                    val_loss = criterion(pred.view(labels.size(dim=0), n_classes, 256, 256), labels.long()) \
                               + dice_loss(F.softmax(pred, dim=1).float(),
                                           F.one_hot(labels.long(), n_classes).permute(0, 3, 1, 2).float(),
                                           multiclass=True)

                    mask_true = F.one_hot(labels.long(), n_classes).permute(0, 3, 1, 2).float()
                    mask_pred = F.one_hot(F.softmax(pred, dim=1).argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
                    val_dice_score += multiclass_dice_coeff(mask_pred[:, :, ...], mask_true[:, :, ...],
                                                              reduce_batch_first=True)

                    tmp_per_class = per_class_dice_coeff(mask_pred[:, :, ...], mask_true[:, :, ...],
                                                         reduce_batch_first=True)

                    for ind in range(len(tmp_per_class)):
                        val_multi_dice_score[ind] += tmp_per_class[ind]


                pred = torch.argmax(F.softmax(pred.view(labels.size(dim=0), n_classes, 256, 256), dim=1).float(), dim=1)
                pred = pred.long().cpu().numpy().astype(np.uint8)

                weight = np.where(labels.cpu().numpy() == 255, 0, 1).astype(bool)
                accuracy = labels.cpu().numpy() == pred
                accuracy = accuracy[weight]
                accuracy = accuracy.sum() / accuracy.size


                val_loss_total += val_loss.item()
                val_accuracy_total += accuracy

                pbar.update(1)

        class_dice = {}
        for c in range(len([x/len(val_loader) for x in val_multi_dice_score])):
            class_dice[f'class_{c}_dice'] = [x/len(val_loader) for x in val_multi_dice_score][c]

        if experiment is not None:
            experiment.log({
                'train epoch loss': train_loss_total / len(train_loader),
                'val epoch loss': val_loss_total / len(val_loader),
                'learning rate': optimizer.param_groups[0]['lr'],
                'train epoch accuracy': train_accuracy_total / len(train_loader),
                'val epoch accuracy': val_accuracy_total / len(val_loader),
                'train epoch dice': train_dice_score / len(train_loader),
                'val epoch dice': val_dice_score / len(val_loader),
                'per class Dice - val': class_dice,
                'epoch': e,
                **histograms
            })

        print(f"Epoch {e} val loss: {val_loss_total/len(val_loader)}")
        print(f"Epoch {e} val accuracy: {val_accuracy_total/len(val_loader)}")
        print(f"Epoch {e} val dice: {val_dice_score/len(val_loader)}")
        print(f"Epoch {e} val class dice: {class_dice}")
        print("---------------------------------")

        if val_loss_total/len(val_loader) < prev_val_loss:
            prev_val_loss = val_loss_total/len(val_loader)
            early_stopping = 0
            torch.save(model.state_dict(), f'./trained_models/{name}.pt')
        else:
            early_stopping += 1

        if early_stopping >= 5:
            print("Early stopping triggered")
            model.load_state_dict(torch.load(f'./trained_models/{name}.pt'))
            break

        ### VALIDATION END #####
    wandb.finish()


if __name__ == '__main__':
    train_model()
