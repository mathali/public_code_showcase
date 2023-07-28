from utils.store_predictions import store_predictions
from unet import UNet
import torch
from torch.utils.data import DataLoader
from utils.data_loading import BasicDataset

# Specify data path of data that you want the model to evaluate and store, this script should take care of the rest.
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    net = UNet(n_channels=3, n_classes=1, bilinear=False)
    net.load_state_dict(torch.load('./outputs/CoNIC_c1_e30_b32_lr0.0003_s1.0_normalizer=macenko/checkpoint_epoch30.pth',
                                   map_location=torch.device('cuda')))
    net.to(device=torch.device('cuda'))

    val_set = BasicDataset('DATA-PATH', 1.0, normalizer="macenko", augment=False, mode='train')
    val_set.set_mode('train')

    loader_args = dict(batch_size=32, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    store_predictions(net, val_loader, torch.device('cuda'), 'OUTPUT-PATH')