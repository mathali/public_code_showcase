import torch.cuda
import wandb
import argparse
import os

from configuration import configuration
from presplit_train import train_net
from model.classifier import Classifier
from model.inception_classifier import InceptionClassifier
from model.dense_classifier import DenseClassifier


"""
Driver responsible for the training process of the nucleus segmentation model.
Fill out training parameters in config.json
Pass metadata (wandb key, data path, model path, log) as python script arguments. 
    Data path can also be loaded from config, model path is optional if you want to resume training
    wandb is forced as an argument, so you don't store your key anywhere
    log switch is used to turn on wandb logging
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', type=str, help='wandb key')
    parser.add_argument('--data_path', type=str, help='path to data')
    parser.add_argument('--load', type=str, help='path to saved model', default=None)
    parser.add_argument('--log', type=bool, help='switch to turn on wandb logging', default=False)
    args = parser.parse_args()

    config = configuration('config.json')
    config.add('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {config.device} device')

    # Initialize logging
    experiment = None
    if args.log:
        wandb.login(key=args.wandb)
        experiment = wandb.init(project='CoNIC--Classification', resume='allow', name=f'SpringNet_Presplit_no_background',
                                config={
                                    "learning_rate": config.learning_rate,
                                    "batch_size": config.batch_size,
                                    "epochs": config.epochs,
                                    "val_percent": config.val_percent,
                                    "save_checkpoint": config.save_checkpoint,
                                    "amp": config.amp,
                                    "l1_lambda": config.l1_lambda,
                                    "l2_lambda": config.l2_lambda
                                    }
                                )
    data = args.data_path
    if data is None:
        data = "../segmentation_postprocessing/patch_datasets/watershed_presplit_balanced"

    print("===== DATA =====")
    print("DATA PATH: " + data)
    print("================")

    models_root = f'{config.dir_checkpoint}/SpringNet_Presplit_no_background'
    if not os.path.exists(models_root):
        os.makedirs(models_root)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config.model_type == 'classic':
        net = Classifier(n_channels=3, n_classes=config.n_classes)
    elif config.model_type == 'dense':
        net = DenseClassifier(n_channels=3, n_classes=config.n_classes)
    elif config.model_type == 'inception':
        net = InceptionClassifier(n_channels=3, n_classes=config.n_classes)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
    net.to(device=device)

    print(f'Number of parameters: {sum(p.numel() for p in net.parameters())}')
    print(f'Number of trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}')

    try:
        train_net(net=net,
                  epochs=config.epochs,
                  batch_size=config.batch_size,
                  learning_rate=config.learning_rate,
                  device=device,
                  data_dir=data,
                  dir_checkpoint=models_root,
                  amp=config.amp,
                  experiment=experiment,
                  l1_lambda=config.l1_lambda)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')

    if args.log:
        wandb.finish()
