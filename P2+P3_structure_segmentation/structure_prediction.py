import time
import torch
import torch.nn.functional as F
import numpy as np
import gc
import cv2
import joblib

from utils.smooth_tiled_predictions import predict_img_with_smooth_windowing
from models.unet_model import UNet
from glob import glob
from configuration import configuration

if __name__ == '__main__':
    """
    Script responsible for running inference on WSIs
    
    Handles all the necessary data processing and full tissue predictions.
    """

    config = configuration('./configs/structure_config.json')
    n_classes = config.n_classes
    mode = config.mode
    norm = config.norm

    downscaled = True
    # name = f"Structure_UNet_{mode}_downscaled_cells_gnorm"
    name = f"{mode.capitalize()}_cells_LAB_{norm}_scaled_diceloss"

    test_wsis = glob('./test_data/cell_src/*.npy')

    device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")
    state_dict = torch.load(f"./trained_models/{name}.pt")

    del_keys = []
    for key in state_dict:
        if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
            del_keys.append(key)

    for key in del_keys:
        del state_dict[key]

    if mode == 'binary':
        model = UNet(n_channels=5, n_classes=n_classes, norm=norm)
    elif mode == 'multiclass':
        model = UNet(n_channels=9, n_classes=n_classes, norm=norm)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    for idx, wsi_path in enumerate(test_wsis):
        out_name = wsi_path.split('\\')[-1]
        start_time = time.time()
        img = np.load(wsi_path)
        org = img[4, :, :]
        scalers = []

        # Binary mode - prepare data that contains the WSI in LAB + binary cell distribution mask + dst transform of said
        # cell mask
        if mode == 'binary':
            scalers.append(joblib.load('./scalers/multiclass/L_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multiclass/A_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multiclass/B_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multiclass/binary_dst_scaler.joblib'))

            wsi = np.zeros((5, img.shape[1], img.shape[2]))
            wsi[:3, :, :] = img[:3, :, :]
            wsi[3, :, :] = np.where(img[4, :, :] > 0, 1, 0)
            dst_data = np.where(img[4, :, :] >= 1, 0, 1).astype(np.uint8)
            dst = cv2.distanceTransform(dst_data, cv2.DIST_L2, 3)
            dst = np.where(dst>255, 255, dst)
            wsi[4, :, :] = dst

            wsi[0] = scalers[0].transform(wsi[0].reshape(-1, 1)).reshape(wsi.shape[1], wsi.shape[2])
            wsi[1] = scalers[1].transform(wsi[1].reshape(-1, 1)).reshape(wsi.shape[1], wsi.shape[2])
            wsi[2] = scalers[2].transform(wsi[2].reshape(-1, 1)).reshape(wsi.shape[1], wsi.shape[2])
            wsi[4] = scalers[3].transform(wsi[4].reshape(-1, 1)).reshape(wsi.shape[1], wsi.shape[2])

            if downscaled:
                wsi[3, :, :] = np.where(wsi[3, :, :] == 1, 255, 0)
                wsi = cv2.resize(wsi.transpose(1, 2, 0), (wsi.shape[2]//2, wsi.shape[1]//2)).transpose(2, 0, 1)

            del dst, dst_data, img

        # Multiclass mode - prepare data that contains the WSI in LAB + multiclass cell distribution mask + dst transform of said
        # cell mask. The final array contains 9 channels - LAB + 2 channels for each cell class
        elif mode == 'multiclass':
            scalers.append(joblib.load('./scalers/multiclass/L_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multiclass/A_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multiclass/B_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multiclass/multiclass_dst1_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multiclass/multiclass_dst2_scaler.joblib'))
            scalers.append(joblib.load('./scalers/multiclass/multiclass_dst3_scaler.joblib'))

            wsi = np.zeros((9, img.shape[1], img.shape[2]))
            wsi[:3, :, :] = img[:3, :, :]
            wsi[3, :, :] = np.where(img[4, :, :] == 1, 1, 0)
            wsi[5, :, :] = np.where(img[4, :, :] == 2, 1, 0)
            wsi[7, :, :] = np.where(img[4, :, :] == 3, 1, 0)

            dst_data_1 = np.where(img[4, :, :] == 1, 0, 1).astype(np.uint8)
            dst_data_2 = np.where(img[4, :, :] == 2, 0, 1).astype(np.uint8)
            dst_data_3 = np.where(img[4, :, :] == 3, 0, 1).astype(np.uint8)

            dst_1 = cv2.distanceTransform(dst_data_1, cv2.DIST_L2, 3)
            dst_2 = cv2.distanceTransform(dst_data_2, cv2.DIST_L2, 3)
            dst_3 = cv2.distanceTransform(dst_data_3, cv2.DIST_L2, 3)

            dst_1 = np.where(dst_1>255, 255, dst_1)
            dst_2 = np.where(dst_2>255, 255, dst_2)
            dst_3 = np.where(dst_3>255, 255, dst_3)

            wsi[4, :, :] = dst_1
            wsi[6, :, :] = dst_2
            wsi[8, :, :] = dst_3

            wsi[0] = scalers[0].transform(wsi[0].reshape(-1, 1)).reshape(wsi.shape[1], wsi.shape[2])
            wsi[1] = scalers[1].transform(wsi[1].reshape(-1, 1)).reshape(wsi.shape[1], wsi.shape[2])
            wsi[2] = scalers[2].transform(wsi[2].reshape(-1, 1)).reshape(wsi.shape[1], wsi.shape[2])
            wsi[4] = scalers[3].transform(wsi[4].reshape(-1, 1)).reshape(wsi.shape[1], wsi.shape[2])
            wsi[6] = scalers[4].transform(wsi[6].reshape(-1, 1)).reshape(wsi.shape[1], wsi.shape[2])
            wsi[8] = scalers[5].transform(wsi[8].reshape(-1, 1)).reshape(wsi.shape[1], wsi.shape[2])

            if downscaled:
                wsi[3, :, :] = np.where(wsi[3, :, :] == 1, 255, 0)
                wsi[5, :, :] = np.where(wsi[5, :, :] == 1, 255, 0)
                wsi[7, :, :] = np.where(wsi[7, :, :] == 1, 255, 0)
                wsi = cv2.resize(wsi.transpose(1, 2, 0), (wsi.shape[2]//2, wsi.shape[1]//2)).transpose(2, 0, 1)

            del dst_1, dst_2, dst_3, dst_data_1, dst_data_2, dst_data_3, img

        gc.collect()

        wsi = np.expand_dims(wsi, axis=0)

        # Automatic prediction on large images
        # Automatically subdivides the image into patches with predefined size and overlap
        # Predicts with redundancy, merges the prediction using a Gaussian window
        pred_combined = predict_img_with_smooth_windowing(wsi,
                                                          window_size=256,
                                                          subdivisions=2,
                                                          nb_classes=n_classes,
                                                          mode="binary",
                                                          pred_func=(
                                                              lambda img_batch_subdiv: F.softmax(model(img_batch_subdiv).view(img_batch_subdiv.size(dim=0), n_classes, 256, 256), dim=1).float()
                                                          ))
        print(f"Execution time: {(time.time() - start_time)/60} min")

        with open(f"./output/tmp/old_{out_name.replace('.tif', '.npy')}", "wb") as f:
            np.save(f, pred_combined)


        del wsi, pred_combined
        gc.collect()


