import time
import torch
import torch.nn.functional as F
import numpy as np
import gc

from utils.smooth_tiled_predictions import predict_img_with_smooth_windowing
from models.unet_model import UNet
from matplotlib import pyplot as plt
from glob import glob

if __name__ == '__main__':
    """
    Runs inference on testing data using the best model. 
    Saves predictions as numpy arrays, so they can be used later.
    """
    n_classes = 4
    mode = "multiclass"

    test_wsis = glob(f'../test_data/cell/multiclass/*.npy')
    device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")
    state_dict = torch.load(f"./trained_models/Nuclei_UNet_{mode}_balanced.pt")
    model = UNet(n_channels=3, n_classes=n_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    for idx, wsi_path in enumerate(test_wsis):
        plt.figure(idx)
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        out_name = wsi_path.split('-')[0].replace("../test_data/cell/multiclass\\", "")
        start_time = time.time()

        org_img = np.load(wsi_path)
        wsi = org_img[:3, :, :]
        wsi = np.expand_dims(wsi, axis=0)
        pred_combined = predict_img_with_smooth_windowing(wsi/255,
                                                          window_size=256,
                                                          subdivisions=2,
                                                          nb_classes=n_classes,
                                                          pred_func=(
                                                              lambda img_batch_subdiv: F.softmax(model(img_batch_subdiv).view(img_batch_subdiv.size(dim=0), n_classes, 256, 256), dim=1).float()
                                                          ))
        print(f"Execution time: {(time.time() - start_time)/60} min")


        pred = np.argmax(pred_combined, axis=0).astype(np.uint8)
        with open(f"../test_data/cell/nuclei_multiclass/{out_name.replace('.tif', '.npy')}", "wb") as f:
            org_img[5, :, :] = pred
            np.save(f, org_img)

        ax[0].imshow(np.transpose(np.squeeze(wsi), (1, 2, 0)))
        ax[0].set_title(str(wsi_path.split('.')[0]) + " WSI")
        ax[1].imshow(pred)
        ax[1].set_title(str(wsi_path.split('.')[0]) + " pred")

        del wsi, pred_combined, pred
        gc.collect()

    plt.show()

