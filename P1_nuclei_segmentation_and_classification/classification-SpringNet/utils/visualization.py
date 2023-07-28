import cv2 as cv
import numpy as np
import glob
from matplotlib import pyplot as plt


def plot_dist(file_list):
    """
    Analyzes the distribution of classes for the used dataset.
    """
    c_dict = np.zeros(7, dtype=np.uint32)
    for file in file_list:
        c = file.split('class=')[1].split('_')[0]
        c_dict[int(c)] += 1

    font = {'family': 'serif',
            'color': 'black',
            'weight': 'bold',
            'size': 16,
            }
    plt.bar([0, 1, 2, 3, 4, 5, 6], np.sort((c_dict / c_dict.sum()))[::-1])
    plt.xticks([0, 1, 2, 3, 4, 5, 6], ['Epithelial', 'Lymphocyte', 'Çonnective', 'Background', 'Plasma', 'Eosinophil', 'Neutrophil'], rotation=45)
    plt.ylabel('Proportion', fontdict=font)
    plt.xlabel('Category', fontdict=font)
    plt.tight_layout()
    plt.show()


def plot_imgs(file_list):
    """
    Provides sample visualizations of the used dataset.
    """
    for file in file_list:
        if file.split('class=')[1].split('_')[0] == '0':
            img = np.load(file).astype(np.uint8)
            img = cv.resize(img, (200, 200))
            cv.imshow(file.split('\\')[1], cv.cvtColor(img, cv.COLOR_RGB2BGR))
            cv.waitKey()
            cv.destroyAllWindows()

if __name__ == '__main__':
    file_list = glob.glob('../../segmentation_postprocessing/patch_datasets/watershed_presplit_balanced/**/*.npy', recursive=True)
    # plot_dist(file_list)
    plot_imgs(file_list)
