import seaborn as sn
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, cohen_kappa_score
from model.classifier import Classifier
from torch.utils.data import DataLoader
from data_loading import NucleiDataset


def store_test_results(test_loader, best_model, device):
    """
    Runs an evalution loop on the the provided dataloader, stores the results so they can be analyzed later.
    """
    best_model.eval()
    with torch.no_grad():
        val_accuracy_total, val_recall_weighted_total, val_precision_weighted_total, f1_weighted_total = 0, 0, 0, 0
        val_loss_total, val_recall_macro_total, val_precision_macro_total, f1_macro_total = 0, 0, 0, 0

        class_mappings = {0: 'Background', 1: 'Connective tissue', 2: 'Eosinophil', 3: 'Epithelial',
                          4: 'Lymhocyte', 5: 'Neutrophil', 6: 'Plasma'}

        count = 0
        print('Initiating prediction')
        for batch in test_loader:
            images = batch['image']
            classes = batch['class']

            images = images.to(device=device, dtype=torch.float32)
            classes = classes.to(device=device, dtype=torch.uint8)

            classes_pred = best_model(images)

            classes_pred = np.argmax(torch.softmax(classes_pred, dim=1).data.float().cpu().numpy(), axis=1)

            if count == 0:
                pred, true = classes_pred, classes
                count += 1
            else:
                pred = np.concatenate((pred, classes_pred), axis=0)
                true = torch.cat([true, classes])
                count += 1

            val_accuracy_total += accuracy_score(classes.data.cpu(), classes_pred)
            val_recall_weighted_total += recall_score(classes.data.cpu(), classes_pred, average="weighted")
            val_precision_weighted_total += precision_score(classes.data.cpu(), classes_pred, average="weighted")
            f1_weighted_total += f1_score(classes.data.cpu(), classes_pred, average="weighted")
            val_recall_macro_total += recall_score(classes.data.cpu(), classes_pred, average="macro")
            val_precision_macro_total += precision_score(classes.data.cpu(), classes_pred, average="macro")
            f1_macro_total += f1_score(classes.data.cpu(), classes_pred, average="macro")

            img_count = 0
            for img, true, pred in zip(images.cpu().detach().numpy(),
                                        classes.cpu().detach().numpy(),
                                        classes_pred):
                np.save(f"../outputs/test_results/batch={count}_image={img_count}_predicted={class_mappings[pred]}_labeled={class_mappings[true]}.npy", img)
                img_count += 1

        print(f'Acc: {val_accuracy_total / len(test_loader):.3f} \
                | F1 weighted: {f1_weighted_total / len(test_loader):.3f}| Prec weighted: {val_precision_weighted_total / len(test_loader):.3f}| \
                 Rec weighted: {val_recall_weighted_total / len(test_loader):.3f}\
                | F1 macro: {f1_macro_total / len(test_loader):.3f}| Prec macro: {val_precision_macro_total / len(test_loader):.3f}| \
                 Rec macro: {val_recall_macro_total / len(test_loader):.3f}')

        full_matrix = confusion_matrix(true.cpu().detach().numpy(), pred)
        print(sum(full_matrix[1:, 1:].diagonal()) / sum(full_matrix[1:, 1:].sum(axis=0)))
        print(f'Kappa with background: {cohen_kappa_score(true.cpu().detach().numpy(), pred)}')
        print(f'Kappa without background: {cohen_kappa_score(true.cpu().detach().numpy(), pred, labels=[1, 2, 3, 4, 5, 6])}')
        cm = confusion_matrix(true.cpu().detach().numpy(), pred, normalize='true')
        sn.set(font_scale=2)
        sn.heatmap(cm, xticklabels=['Background', 'Connective\ntissue', 'Eosinophil', 'Epithelial', 'Lymphocyte', 'Neutrophil', 'Plasma'],
                   yticklabels=['Background', 'Connective\ntissue', 'Eosinophil', 'Epithelial', 'Lymphocyte', 'Neutrophil', 'Plasma'],
                   annot_kws={"size": 24}, annot=True, linewidths=0.5, cmap=sn.color_palette("Blues", as_cmap=True), cbar=False)
        plt.xticks(rotation=40)
        plt.xlabel('Predicted class', fontdict={'weight': 'bold', 'size': 24}, labelpad=0)
        plt.ylabel('True class', fontdict={'weight': 'bold', 'size': 24})
        plt.rcParams.update({'font.size': 72, 'font.weight': 'bold', 'figure.autolayout': True})
        # plt.savefig(fname='confusion_matrix.png', format='png', bbox_inches='tight')
        plt.show()




if __name__ == '__main__':
    test_path = "../segmentation_postprocessing/patch_datasets/watershed_presplit_balanced/test"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    net = Classifier(n_channels=3, n_classes=7)
    net.load_state_dict(torch.load(
        '../outputs/SpringNet-no_overlap-watershed-balanced/checkpoint_epoch28.pth',
        map_location=torch.device('cuda')))
    net.to(device=torch.device('cuda'))
    print('Model loaded')

    test_set = NucleiDataset("TEST-SET-PATH", mode='test')
    print('Dataset prepared')

    test_loader = DataLoader(test_set, shuffle=True, drop_last=False, batch_size=8192, num_workers=0)
    print('Loader prepared')

    store_test_results(test_loader, net, torch.device('cuda'))