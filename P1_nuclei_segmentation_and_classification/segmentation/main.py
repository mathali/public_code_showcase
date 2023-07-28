import torch.cuda
import wandb
import argparse
import os
import logging

from configuration import configuration
from train import train_net
from unet import UNet

"""
Driver responsible for the training process of the nucleus segmentation model.
Fill out training parameters in config.json
Pass metadata (wandb key, data path, model path) as python script arguments. 
    Data path can also be loaded from config, model path is optional if you want to resume training
    wandb is forced as an argument, so you don't store your key anywhere
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', type=str, help='wandb key')
    parser.add_argument('--data_path', type=str, help='path to data')
    parser.add_argument('--load', type=str, help='path to saved model', default=None)
    args = parser.parse_args()

    config = configuration('config.json')
    config.add('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {config.device} device')

    wandb.login(key=args.wandb)
    experiment = None

    # Initialize logging
    # experiment = wandb.init(project='CoNIC--U-Net', resume='allow', name=f'pre-normalized__b_size={config.batch_size}_{config.learning_rate}_e{config.epochs}_c{config.n_classes}_normalizer={config.normalizer}_augment={config.augment}',
    #                         config={
    #                             "learning_rate": config.learning_rate,
    #                             "batch_size": config.batch_size,
    #                             "epochs": config.epochs,
    #                             "save_checkpoint": config.save_checkpoint,
    #                             "img_scale": config.img_scale,
    #                             "amp": config.amp,
    #                             }
    #                         )

    data = args.data_path

    # Data can also be loaded from config
    if data == None:
        data = config.data_path

    print("===== DATA =====")
    print("DATA PATH: " + data)
    print("================")

    models_root = f'{config.dir_checkpoint}/{config.dataset}_c{config.n_classes}_e{config.epochs}_b{config.batch_size}_lr{config.learning_rate}_s{config.img_scale}_normalizer={config.normalizer}'
    if not os.path.exists(models_root):
        os.makedirs(models_root)
    print(os.listdir(os.path.dirname(models_root)))
    print(os.path.abspath(models_root))


    logging.basicConfig(filename='training.log', level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net = UNet(n_channels=3, n_classes=config.n_classes, bilinear=config.bilinear)
    # wandb.watch(net)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')
    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=config.epochs,
                  batch_size=config.batch_size,
                  learning_rate=config.learning_rate,
                  device=device,
                  img_scale=config.img_scale,
                  data_dir=data,
                  dir_checkpoint=models_root,
                  amp=config.amp,
                  experiment=experiment,
                  normalizer=config.normalizer,
                  augment=config.augment)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')

    wandb.finish()
