import cv2 as cv
import numpy as np
import glob
import random
import shutil

def balance(file_list):
    """
    Creates a balanced dataset - undersamples overrepresented classes
    """
    balanced_list = [[], [], [], [], [], [], []]
    for file in file_list:
        label = file.split('class=')[1].split('_')[0]
        balanced_list[int(label)].append(file)

    balanced_list[2] = random.sample(balanced_list[2], 100000)
    balanced_list[3] = random.sample(balanced_list[3], 60000)

    for label in balanced_list:
        for file in label:
            shutil.copy2(file, "./patch_datasets/watershed_balanced/")


def visualize(file_list):
    """
    Visualize samples of the created dataset for validation
    """
    for file in file_list:
        img = np.load(file)

        class_mappings = {0: 'Background', 1: 'Connective tissue', 2: 'Eosinophil', 3: 'Epithelial',
                          4: 'Lymphocyte', 5: 'Neutrophil', 6: 'Plasma'}
        if class_mappings[int(file.split('class=')[-1].split('_')[0])] != 'Connective tissue':
            continue
        print(class_mappings[int(file.split('class=')[-1].split('_')[0])])
        sample = cv.cvtColor(img.astype(np.uint8), cv.COLOR_RGB2BGR)
        sample = cv.resize(sample, (200, 200))
        cv.imshow('sample', sample)
        cv.waitKey()
        cv.destroyAllWindows()


if __name__ == '__main__':
    file_list = glob.glob("./patch_datasets/watershed_presplit_balanced/train/*.npy")

    visualize(file_list)
    # balance(file_list)