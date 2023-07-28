import torch
import torch.nn.functional as F
import numpy as np
import wandb

from tqdm import tqdm
from data_utils.cell_dataset import IKEMCellDataset
from torch.utils.data import DataLoader
from models.unet_model import UNet
from utils.balanced_losses import MiniBatchCrossEntropyLoss, multiclass_dice_coeff, per_class_dice_coeff, dice_loss
from configuration import configuration


def train_model():
    """ Primary training script
    Responsible for every part of the training process - training, validating, and testing
    Stores model at every checkpoint at which the model's validation performance improves

    Parameters are controlled using cell_config.json
    """

    ### TRAINING SETUP START #####
    config = configuration('./configs/cell_config.json')
    log = config.log
    lr = config.lr
    n_classes = config.n_classes
    val_percent = config.val_percent
    test_percent = config.test_percent
    l2_lambda = config.l2_lambda
    batch_size = config.batch_size
    epochs = config.epochs

    name = "Nuclei_UNet_multiclass_balanced"

    dataset = IKEMCellDataset(data_dir='F:/Halinkovic/IKEM/nuclei_training_data/', mode='multiclass')

    val_size = int(val_percent * len(dataset))
    test_size = int(test_percent * len(dataset))
    train_size = len(dataset) - val_size - test_size

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size],
                                                                 generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=16,
                              pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=16,
                            pin_memory=True, prefetch_factor=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=16,
                            pin_memory=True, prefetch_factor=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=n_classes)
    model.to(device)

    optimizer = torch.optim.Adam([p for p in model.parameters()], lr=lr, weight_decay=l2_lambda)
    criterion = MiniBatchCrossEntropyLoss()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

    model = model.to(device)

    prev_val_loss = np.inf
    early_stopping = 0

    print(f'Total number of trainable parameters: {sum(p.numel() for p in model.parameters())}')

    experiment = None
    if log:
        wandb.login(key="YOUR-KEY-HERE")
        experiment = wandb.init(project='IKEM-Cell-Segmentation', resume='allow', name=name,
                                config={
                                    "learning_rate": lr,
                                    "batch_size": batch_size,
                                    "epochs": epochs,
                                    "val_percent": val_percent,
                                    "test_percent": test_percent,
                                    "l2_lambda": l2_lambda,
                                    "n_classes": n_classes,
                                    "loss_func": "MiniBatchCrossEntropy"
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
                       + dice_loss(F.softmax(pred.view(labels.size(dim=0), 2, 256, 256), dim=1).float(),
                                    F.one_hot(labels.long(), 3).permute(0, 3, 1, 2).float(),
                                    multiclass=True)

                mask_true = F.one_hot(labels.long(), n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(pred.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
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
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                if experiment is not None:
                    experiment.log({
                        'train loss': loss.item(),
                        'train accuracy': accuracy,
                    })

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
                               + dice_loss(F.softmax(pred.view(labels.size(dim=0), 2, 256, 256), dim=1).float(),
                                           F.one_hot(labels.long(), 3).permute(0, 3, 1, 2).float(),
                                           multiclass=True)

                    mask_true = F.one_hot(labels.long(), n_classes).permute(0, 3, 1, 2).float()
                    mask_pred = F.one_hot(pred.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
                    val_dice_score += multiclass_dice_coeff(mask_pred[:, :, ...], mask_true[:, :, ...],
                                                              reduce_batch_first=False)

                    tmp_per_class = per_class_dice_coeff(mask_pred[:, :, ...], mask_true[:, :, ...],
                                                         reduce_batch_first=False)

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
        print("---------------------------------")

        if val_loss_total/len(val_loader) < prev_val_loss:
            prev_val_loss = val_loss_total/len(val_loader)
            early_stopping = 0
            torch.save(model.state_dict(), f'./trained_models/{name}.pt')
        else:
            early_stopping += 1

        if early_stopping >= 3:
            print("Early stopping triggered")
            model.load_state_dict(torch.load(f'./trained_models/{name}.pt'))
            break

        ### VALIDATION END #####

    ### TESTING START #####
    model.eval()
    test_loss_total = 0
    test_accuracy_total = 0
    test_dice_score = 0
    test_multi_dice_score = [0] * n_classes

    with tqdm(total=len(test_loader), desc='Batch') as pbar:
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                pred = model(data)
                val_loss = criterion(pred.view(labels.size(dim=0), n_classes, 256, 256), labels.long()) \
                               + dice_loss(F.softmax(pred.view(labels.size(dim=0), 2, 256, 256), dim=1).float(),
                                           F.one_hot(labels.long(), 3).permute(0, 3, 1, 2).float(),
                                           multiclass=True)

                mask_true = F.one_hot(labels.long(), n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(pred.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
                test_dice_score += multiclass_dice_coeff(mask_pred[:, :, ...], mask_true[:, :, ...],
                                                          reduce_batch_first=False)

                tmp_per_class = per_class_dice_coeff(mask_pred[:, :, ...], mask_true[:, :, ...],
                                                     reduce_batch_first=False)

                for ind in range(len(tmp_per_class)):
                    test_multi_dice_score[ind] += tmp_per_class[ind]

                pred = torch.argmax(F.softmax(pred.view(labels.size(dim=0), n_classes, 256, 256), dim=1).float(), dim=1)
                pred = pred.long().cpu().numpy().astype(np.uint8)

                weight = np.where(labels.cpu().numpy() == 255, 0, 1).astype(bool)
                accuracy = labels.cpu().numpy() == pred
                accuracy = accuracy[weight]
                accuracy = accuracy.sum() / accuracy.size

            test_loss_total += val_loss.item()
            test_accuracy_total += accuracy

            pbar.update(1)

    print(f"Test loss: {test_loss_total / len(test_loader)}")
    print(f"Test accuracy: {test_accuracy_total / len(test_loader)}")
    print(f"Test dice: {test_dice_score / len(test_loader)}")
    if experiment is not None:
        class_dice = {}
        for c in range(len([x/len(val_loader) for x in test_multi_dice_score])):
            class_dice[f'class_{c}_dice'] = [x/len(val_loader) for x in test_multi_dice_score][c]
        experiment.log({
            'Test loss': test_loss_total / len(test_loader),
            'Test accuracy': test_accuracy_total / len(test_loader),
            'Test dice': test_dice_score / len(test_loader),
            'per class Dice - test': class_dice,
        })
        wandb.finish()

    ### TESTING END #####

if __name__ == '__main__':
    train_model()
