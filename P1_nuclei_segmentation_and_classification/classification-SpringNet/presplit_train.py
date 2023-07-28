import torch
import torch.nn as nn
import wandb
import numpy as np

from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pathlib import Path
from utils.data_loading import NucleiDataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def train_net(net,
              device,
              epochs: int = 50,
              batch_size: int = 32,
              learning_rate: float = 1e-4,
              save_checkpoint: bool = True,
              amp: bool = False,
              data_dir: str = '',
              dir_checkpoint: str = 'outputs',
              experiment = None,
              l1_lambda: float = 0):

    """ Primary training script
    Responsible for all training steps - training, validation and testing
    Stores model at every checkpoint at which the model's validation performance improves
    Training contains class weight balancing and L1 regularization to help prevent overfitting due to the data imbalance
    of the Lizard dataset.

    Parameters are controlled using config.json
    """

    ### TRAINING SETUP START #####
    train_set = NucleiDataset(data_dir+'/train', mode='train')
    val_set = NucleiDataset(data_dir+'/valid',mode='val')
    test_set = NucleiDataset(data_dir+'/test', mode='test')

    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, batch_size=batch_size, num_workers=4)
    val_loader = DataLoader(val_set, shuffle=True, drop_last=False, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_set, shuffle=True, drop_last=False, batch_size=batch_size, num_workers=4)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    if net.n_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        # Precalculated class weights to balance the data contained in Lizard
        class_weights = torch.tensor([0.90269845, 2.12959964, 0.64620571, 0.64620571, 1.92380385,  3.84097548, 0.64620571], dtype=torch.float)

        class_weights = class_weights.to(device=device, dtype=torch.float)
        criterion = nn.CrossEntropyLoss(class_weights)
    global_step = 0

    ### TRAINING SETUP END #####



    ### MODEL TRAINING START #####
    prev_best = np.inf
    for epoch in range(epochs):
        train_loss_total = 0
        val_loss_total = 0
        net.train()

        ### TRAINING START #####
        with tqdm(total=len(train_set), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                classes = batch['class']

                images = images.to(device=device, dtype=torch.float32)
                classes = classes.to(device=device, dtype=torch.uint8)

                with torch.cuda.amp.autocast(enabled=amp):
                    classes_pred = net(images)

                    if net.n_classes == 1:
                        classes_pred = torch.squeeze(classes_pred)
                        loss = criterion(classes_pred, classes.float())

                    else:
                        loss = criterion(classes_pred, classes)


                l1_norm = 0
                for layer in [net.small_spring_1, net.small_spring_2, net.up_1, net.medium_spring_1,
                              net.medium_spring_2, net.up_2, net.fc1]:
                    l1_norm += l1_lambda * sum(p.abs().sum() for p in layer.parameters())

                loss += l1_norm

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                train_loss_total += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})


            histograms = {}
            for tag, value in net.named_parameters():
                tag = tag.replace('/', '.')
                if value.grad is not None and experiment is not None:
                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

            ### TRAINING END #####

            ### VALIDATION START #####
            net.eval()
            with tqdm(total=len(val_set), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                count = 0
                with torch.no_grad():
                    val_accuracy_total, val_recall_weighted_total, val_precision_weighted_total, f1_weighted_total = 0, 0, 0, 0
                    val_recall_macro_total, val_precision_macro_total, f1_macro_total = 0, 0, 0
                    for batch in val_loader:
                        images = batch['image']
                        classes = batch['class']

                        images = images.to(device=device, dtype=torch.float32)
                        classes = classes.to(device=device, dtype=torch.uint8)

                        classes_pred = net(images)

                        l1_norm = 0
                        for layer in [net.small_spring_1, net.small_spring_2, net.up_1, net.medium_spring_1,
                                      net.medium_spring_2, net.up_2, net.fc1]:
                            l1_norm += l1_lambda * sum(p.abs().sum() for p in layer.parameters())

                        val_loss = criterion(classes_pred, classes) + l1_norm
                        val_loss_total += val_loss.item()

                        classes_pred = np.argmax(torch.softmax(classes_pred, dim=1).data.float().cpu().numpy(), axis=1)
                        if count == 0:
                            pred, true = classes_pred, classes
                            count += 1
                        else:
                            pred = np.concatenate((pred, classes_pred), axis=0)
                            true = torch.cat([true, classes])

                        val_accuracy_total += accuracy_score(classes.data.cpu(), classes_pred)
                        val_recall_weighted_total += recall_score(classes.data.cpu(), classes_pred, average="weighted")
                        val_precision_weighted_total += precision_score(classes.data.cpu(), classes_pred, average="weighted")
                        f1_weighted_total += f1_score(classes.data.cpu(), classes_pred, average="weighted")
                        val_recall_macro_total += recall_score(classes.data.cpu(), classes_pred, average="macro")
                        val_precision_macro_total += precision_score(classes.data.cpu(), classes_pred, average="macro")
                        f1_macro_total += f1_score(classes.data.cpu(), classes_pred, average="macro")

            if experiment is not None:
                experiment.log({
                    'train loss': train_loss_total/len(train_loader),
                    'val loss': val_loss_total/len(val_loader),
                    'learning rate': optimizer.param_groups[0]['lr'],
                    'val accuracy': val_accuracy_total/len(val_loader),
                    'val weighted recall': val_recall_weighted_total/len(val_loader),
                    'val weighted precision': val_precision_weighted_total/len(val_loader),
                    'val weighted f1': f1_weighted_total/len(val_loader),
                    'val macro recall': val_recall_macro_total/len(val_loader),
                    'val macro precision': val_precision_macro_total/len(val_loader),
                    'val macro f1': f1_macro_total/len(val_loader),
                    'step': global_step,
                    'epoch': epoch,
                    "val_conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                                preds=pred, y_true=true.data.cpu().numpy(),
                                                                class_names=['Background', 'Eosinophil', 'Epithelial',
                                                                             'Lymphocyte', 'Plasma', 'Neutrophil',
                                                                             'Connective tissue']),
                    **histograms
                })

        scheduler.step()

        if save_checkpoint and val_loss < prev_best:
            prev_best = val_loss
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            best_model = net
            torch.save(net.state_dict(), f'{dir_checkpoint}/checkpoint_epoch{epoch+1}.pth')

        ### VALIDATION END #####

    ### MODEL TRAINING END #####


    ### MODEL TESTING START #####
    best_model.eval()

    with torch.no_grad():
        test_accuracy_total, test_recall_weighted_total, test_precision_weighted_total, f1_weighted_total = 0, 0, 0, 0
        test_loss_total, test_recall_macro_total, test_precision_macro_total, f1_macro_total = 0, 0, 0, 0

        count = 0
        for batch in test_loader:
            images = batch['image']
            classes = batch['class']

            images = images.to(device=device, dtype=torch.float32)
            classes = classes.to(device=device, dtype=torch.uint8)

            classes_pred = best_model(images)

            test_loss = criterion(classes_pred, classes)
            test_loss_total += test_loss.item()


            classes_pred = np.argmax(torch.softmax(classes_pred, dim=1).data.float().cpu().numpy(), axis=1)
            if count == 0:
                pred, true = classes_pred, classes
                count += 1
            else:
                pred = np.concatenate((pred, classes_pred), axis=0)
                true = torch.cat([true, classes])

            test_accuracy_total += accuracy_score(classes.data.cpu(), classes_pred)
            test_recall_weighted_total += recall_score(classes.data.cpu(), classes_pred, average="weighted")
            test_precision_weighted_total += precision_score(classes.data.cpu(), classes_pred, average="weighted")
            f1_weighted_total += f1_score(classes.data.cpu(), classes_pred, average="weighted")
            test_recall_macro_total += recall_score(classes.data.cpu(), classes_pred, average="macro")
            test_precision_macro_total += precision_score(classes.data.cpu(), classes_pred, average="macro")
            f1_macro_total += f1_score(classes.data.cpu(), classes_pred, average="macro")

            if experiment is not None:
                experiment.log({"test_conf_mat" : wandb.plot.confusion_matrix(probs=None,
                                preds=pred, y_true=true.data.cpu().numpy(),
                                class_names=['Background', 'Eosinophil', 'Epithelial', 'Lymphocyte', 'Plasma', 'Neutrophil', 'Connective tissue'])})

    ### MODEL TESTING END #####
