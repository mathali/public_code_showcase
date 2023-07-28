import cv2 as cv
import numpy as np
import glob
import pandas as pd
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight


def process_contour(x, y, h, w, mask, img, classes, count, file):
    """
    Handles the creation of the 15x15px image patches.
    With mirror padding.
    """
    start_y, end_y = max((y + h // 2) - 7, 0), min((y + h // 2) + 8, 255)
    start_x, end_x = max((x + w // 2) - 7, 0), min((x + w // 2) + 8, 255)

    len_y = end_y - start_y
    len_x = end_x - start_x

    mask[start_y:end_y, start_x:end_x] = 0
    patch = img[start_y:end_y, start_x:end_x, :]
    sample = cv.cvtColor(patch.astype(np.uint8), cv.COLOR_RGB2BGR)
    counts = np.unique(classes[(y + h // 2) - 7:(y + h // 2) + 8, (x + w // 2) - 7:(x + w // 2) + 8],
                       return_counts=True)

    if len_y < 15:
        if start_y == 0:
            padding = sample[:15-len_y, :, :]
            padding = np.flip(padding, axis=0)
            sample = np.concatenate((padding, sample))
        elif end_y == 255:
            padding = sample[(len_y)-(15-len_y):, :, :]
            padding = np.flip(padding, axis=0)
            sample = np.concatenate((sample, padding))

    if len_x < 15:
        if start_x == 0:
            padding = sample[:, :15-len_x, :]
            padding = np.flip(padding, axis=1)
            sample = np.concatenate((padding, sample), axis=1)
        elif end_x == 255:
            padding = sample[:, (len_x)-(15-len_x):, :]
            padding = np.flip(padding, axis=1)
            sample = np.concatenate((sample, padding), axis=1)


    label = 0
    for l, c in zip(counts[0], counts[1]):
        if l != 0:
            label = l
            break

    file = file.split('=')[-1][:-4]
    sample = cv.resize(sample, (200, 200))
    cv.imshow('sample', sample)
    cv.waitKey()
    cv.destroyAllWindows()
    np.save(f'./output/file={file}_class={int(label)}_y={(y + h // 2)}_x={(x + w // 2)}', sample)
    return mask


def process_contour_no_mirror(cX, cY, mask, img, classes, count, file):
    """
    Handles the creation of the 15x15px image patches.
    Without mirror padding.
    """
    start_y, end_y = max(cY - 7, 0), min(cY + 8, 255)
    start_x, end_x = max(cX - 7, 0), min(cX + 8, 255)

    len_y = end_y - start_y
    len_x = end_x - start_x

    mask[start_y:end_y, start_x:end_x] = 0
    counts = np.unique(classes[cY - 7:cY + 8, cX - 7:cX + 8],
                       return_counts=True)

    # If unable to extract 15x15 area (because nucleus is located near the edge), shift the center of the patch accordingly
    if len_y < 15:
        if start_y == 0:
            end_y = end_y + (15 - len_y)
        elif end_y == 255:
            start_y = start_y - (15 - len_y)

    if len_x < 15:
        if start_x == 0:
            end_x = end_x + (15 - len_x)
        elif end_x == 255:
            start_x = start_x - (15 - len_x)

    patch = img[start_y:end_y, start_x:end_x, :]
    if patch.shape != (15, 15, 3):
        assert 'Patch offset failed'


    label = 0
    for l, c in zip(counts[0], counts[1]):
        if l != 0:
            label = l
            break

    file = file.split('=')[-1][:-4]
    # sample = cv.cvtColor(patch.astype(np.uint8), cv.COLOR_RGB2BGR)
    # sample = cv.resize(sample, (200, 200), interpolation=cv.INTER_NEAREST)
    # cv.imshow('sample', sample)
    # cv.waitKey()
    # cv.destroyAllWindows()
    # if label not in [0, 2, 3, 4, 6]:
    np.save(f'./patch_datasets/watershed_presplit/test/file={file}_class={int(label)}_cY={cY}_cX={cY}', patch)
    if label in [1, 5]:
        v_patch = cv.flip(patch, 0)
        h_patch = cv.flip(patch, 1)
        vh_patch = cv.flip(patch, -1)
        np.save(f'./patch_datasets/watershed_presplit/train/file={file}_class={int(label)}_cY={cY}_cX={cY}_vertical', v_patch)
        np.save(f'./patch_datasets/watershed_presplit/train/file={file}_class={int(label)}_cY={cY}_cX={cY}_horizontal', h_patch)
        np.save(f'./patch_datasets/watershed_presplit/train/file={file}_class={int(label)}_cY={cY}_cX={cY}_vertical&horizontal', vh_patch)
    return mask


def extract_nuclei(file, global_count):
    """
    Driver for nucleus extraction.
    Preprocesses segmentation masks and extracts individual nuclei based on image moments.

    """
    img = np.load(file)
    mask = img[:, :, 4].astype(np.uint8)
    classes = img[:, :, 5]
    img = img[:, :, :3]

    count = 0
    pre_count = 0
    while 1 in np.unique(mask):
        # Run custom watershed algorithm on mask to help separate closely packed nuclei
        mask = watershed(mask.copy(), img.copy())
        contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        count = len(contours)

        for i in range(0, len(contours)):
            if cv.contourArea(contours[i]) == 0:
                continue

            # Use moments to find center of contour (instead of bounding boxes)
            moments = cv.moments(contours[i])
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            if cv.contourArea(contours[i]) > 10:
                global_count += 1
                mask = process_contour_no_mirror(cX, cY, mask, img, classes, count, file)
            else:
                mask[cY-7:cY+8, cX-7:cX+8] = 0

        if count == pre_count:
            break

        pre_count = count

        kernel = np.ones((3, 3), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        # mask = np.where(mask == 1, 255, mask)
        # cv.imshow('org', mask)
        # cv.waitKey()
        # mask = np.where(mask == 255, 1, mask)

    return 1, global_count


def watershed(mask, img):
    """
    Custom watershed algorithm that combines distance mappings and inverse values of image intensities to improve
    the quality of nuclei segmentation.
    Inverse of intensities is included because the center of nuclei tend to be darker than the rest of the image.
    """

    org = mask.copy()
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)

    sure_bg = cv.dilate(opening, kernel, iterations=2)

    # Euclidean dst transform to get standard watershed distance map
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)

    # Take inverse of nucleus intensities - centers of nuclei tend to be darker
    intensities = np.mean((img * cv.cvtColor(mask, cv.COLOR_GRAY2BGR)).astype(np.uint8), axis=2).astype(np.uint8)
    intensities = np.where(intensities > 0, 255 - intensities, 0)
    intensities = (intensities / intensities.max()) * 5

    # Combine dst map with intensity maps #
    dist_transform += intensities.astype(np.float64)

    # Continue with standard watershed algorithm
    ret, sure_fg = cv.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    ret, markers = cv.connectedComponents(sure_fg)

    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

    markers = cv.watershed(cv.cvtColor(org, cv.COLOR_GRAY2BGR), markers)

    # Delete marks from mask - separates contours of nuclei, patch extractor now works better
    mask[markers == -1] = [0, 0, 0]

    return cv.cvtColor(mask, cv.COLOR_BGR2GRAY).astype(np.uint8)


def driver():
    """
    Extracts the segmentation output of UNet and uses it to create datasets for the classifier.
    """
    global_count = 0
    for split in ['train', 'val', 'test']:
        file_list = glob.glob(f"../segmentation/unet/outputs/CoNIC_c1_e30_b32_lr0.0003_s1.0_normalizer=macenko/no_overlap_{split}_output/*.npy")
        # with Pool(12) as p:
        #     list(tqdm(p.imap(extract_nuclei, file_list), total=len(file_list)))

        # extract_nuclei('F:/Halinkovic/unet/unet/outputs/CoNIC_c1_e30_b32_lr0.0003_s1.0_normalizer=macenko/test_output/batch=11_img=crag_25_000.npy.npy')
        with tqdm(total=len(file_list), desc=f'Nuclei extraction', unit='img') as pbar:
            for file in file_list:
                _, global_count = extract_nuclei(file, global_count)
                pbar.update(1)
    print(global_count)


def balance():
    """
    Function for resampling the training set to correct the imbalance of the Lizard dataset
    """
    file_list = glob.glob("./patch_datasets/watershed_presplit/train" + '/*.npy')

    classes = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0}
    clist = []
    for file in file_list:
        c = file.split('=')[2].split('_')[0]
        clist.append(c)
        classes[c] += 1

    file_df = pd.DataFrame(file_list)
    file_df['class'] = clist
    file_df_2 = file_df[file_df['class']=='2'].sample(50000)
    file_df_3 = file_df[file_df['class']=='3'].sample(50000)
    file_df_6 = file_df[file_df['class']=='6'].sample(50000)
    file_df_1 = file_df[file_df['class']=='1']
    file_df_0 = file_df[file_df['class']=='0']
    file_df_4 = file_df[file_df['class']=="4"]
    file_df_5 = file_df[file_df['class']=='5']

    for df in [file_df_0, file_df_1, file_df_2, file_df_3, file_df_4, file_df_5, file_df_6]:
        for file_name in df[0].values:
            file = np.load(file_name)
            np.save(f"./patch_datasets/watershed_presplit_balanced/{file_name.split('/')[-1]}", file)


def compute_class_weights():
    """
    Precomputes class weights that are later used for cross-entropy when training the classification model.
    """

    file_list = glob.glob("./patch_datasets/watershed_presplit_balanced/train" + '/*.npy')

    classes = {'0':0, '1': 0, '2':0, '3':0, '4':0, '5': 0, '6': 0}
    clist = []
    for file in file_list:
        c = file.split('=')[2].split('_')[0]
        clist.append(c)
        classes[c] += 1

    weight = compute_class_weight(class_weight='balanced',
                                  classes=np.unique(np.asarray(clist, dtype=np.uint8)),
                                  y=np.asarray(clist, dtype=np.uint8))
    print(classes)
    print(weight)


if __name__ == '__main__':
    # balance()
    # driver()
    compute_class_weights()
