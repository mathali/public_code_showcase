import time
import torch
import numpy as np
import gc
import joblib
import cv2

from utils.smooth_tiled_predictions import predict_img_with_smooth_windowing
from models.unet_model import AttentionUNet, PyramidAttentionUNet
from glob import glob
from configuration import configuration

if __name__ == '__main__':
    """
    Script responsible for running inference on WSIs
    
    Handles all the necessary data processing and full wsi predictions.
    """
    # name = "AdditionalData_PyramidAttentionUNet_multiclass_LAB_batchnorm_scaled_BCE+DC"
    name = "PyramidAttentionUNet_multiclass_LAB_batchnorm_scaled_BCE+DC"

    config = configuration('./configs/multilabel_config.json')
    batch_size = config.batch_size
    lr = config.lr
    epochs = config.epochs
    log = config.log
    n_classes = config.n_classes
    mode = config.mode
    norm = config.norm
    nodst = config.nodst
    attention = config.attention
    downscaled = True

    test_wsis = glob('./test_data/multilabel/*.npy')
    data_files = []
    label_files = []
    for file in test_wsis:
        if 'labels' in file:
            label_files.append(file)
        else:
            data_files.append(file)

    device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")
    state_dict = torch.load(f"./trained_models/{name}.pt")

    if mode == 'multiclass_pyramid' and attention:
        model = PyramidAttentionUNet(n_channels=3, n_classes=n_classes)
    elif mode == 'multiclass' and attention:
        model = AttentionUNet(n_channels=3, n_classes=n_classes)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    for idx, paths in enumerate(zip(data_files, label_files)):
        wsi_path, label_path = paths

        out_name = wsi_path.split('\\')[-1]
        start_time = time.time()
        img = np.load(wsi_path)
        org = img[4, :, :]

        # Preprepared training scalers.
        # The WSI is scaled first, the pyramid is generated in the prediction script from the scaled data.
        scalers = []
        scalers.append(joblib.load('./scalers/multilabel/L_scaler.joblib'))
        scalers.append(joblib.load('./scalers/multilabel/A_scaler.joblib'))
        scalers.append(joblib.load('./scalers/multilabel/B_scaler.joblib'))
        scalers.append(joblib.load('./scalers/multilabel/multiclass_dst1_scaler.joblib'))
        scalers.append(joblib.load('./scalers/multilabel/multiclass_dst2_scaler.joblib'))
        scalers.append(joblib.load('./scalers/multilabel/multiclass_dst3_scaler.joblib'))

        wsi = np.zeros((6, img.shape[1], img.shape[2]), dtype=np.float32)
        wsi[:3, :, :] = img[:3, :, :].astype(np.float32) / 255
        wsi[:3, :, :] = cv2.cvtColor(wsi[:3, :, :].transpose(1, 2, 0), cv2.COLOR_RGB2LAB).transpose(2, 0, 1)

        # Preparing the distance transforms that represent cell distributions in the tissue
        dst_data_1 = np.where(img[4, :, :] == 1, 0, 1).astype(np.uint8)
        dst_data_2 = np.where(img[4, :, :] == 2, 0, 1).astype(np.uint8)
        dst_data_3 = np.where(img[4, :, :] == 3, 0, 1).astype(np.uint8)

        dst_1 = cv2.distanceTransform(dst_data_1, cv2.DIST_L2, 3)
        dst_2 = cv2.distanceTransform(dst_data_2, cv2.DIST_L2, 3)
        dst_3 = cv2.distanceTransform(dst_data_3, cv2.DIST_L2, 3)

        dst_1 = np.where(dst_1>255, 255, dst_1)
        dst_2 = np.where(dst_2>255, 255, dst_2)
        dst_3 = np.where(dst_3>255, 255, dst_3)

        wsi[3, :, :] = dst_1
        wsi[4, :, :] = dst_2
        wsi[5, :, :] = dst_3

        if downscaled:
            wsi = cv2.resize(wsi.transpose(1, 2, 0), (wsi.shape[2]//2, wsi.shape[1]//2)).transpose(2, 0, 1)

        wsi[0] = scalers[0].transform(wsi[0].reshape(-1, 1)).reshape(wsi.shape[1], wsi.shape[2])
        wsi[1] = scalers[1].transform(wsi[1].reshape(-1, 1)).reshape(wsi.shape[1], wsi.shape[2])
        wsi[2] = scalers[2].transform(wsi[2].reshape(-1, 1)).reshape(wsi.shape[1], wsi.shape[2])
        wsi[3] = scalers[3].transform(wsi[3].reshape(-1, 1)).reshape(wsi.shape[1], wsi.shape[2])
        wsi[4] = scalers[4].transform(wsi[4].reshape(-1, 1)).reshape(wsi.shape[1], wsi.shape[2])
        wsi[5] = scalers[5].transform(wsi[5].reshape(-1, 1)).reshape(wsi.shape[1], wsi.shape[2])

        del dst_1, dst_2, dst_3, dst_data_1, dst_data_2, dst_data_3, img


        gc.collect()


        wsi = np.expand_dims(wsi, axis=0)
        # Automatic prediction on large images
        # Pyramid context is created during prediction
        # Automatically subdivides the image into patches with predefined size and overlap
        # Predicts with redundancy, merges the prediction using a Gaussian window
        pred_combined = predict_img_with_smooth_windowing(wsi,
                                                          window_size=256,
                                                          subdivisions=2,
                                                          nb_classes=n_classes,
                                                          mode="multiclass_pyramid",
                                                          pred_func=(
                                                              lambda img_batch_subdiv: torch.sigmoid(model(img_batch_subdiv).view(img_batch_subdiv.size(dim=0), n_classes, 256, 256)).float()
                                                          ))
        print(f"Execution time: {(time.time() - start_time)/60} min")

        with open(f"./output/tmp/{out_name.replace('.tif', '.npy')}", "wb") as f:
            np.save(f, pred_combined)

        del wsi, pred_combined
        gc.collect()

