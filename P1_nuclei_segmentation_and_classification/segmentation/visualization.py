import numpy as np
import cv2 as cv
import glob

# Qualitative analysis / Visualization of model performance on stored testing data
if __name__ == '__main__':
    path = './data/test'

    file_list = glob.glob(path + '/*.npy')

    # Overlay predictions - model output in one channel, mask in the other
    for file in file_list:
        img = np.load(file)
        meta = img[:, :, 3:].copy()
        img = img.astype(np.uint8)[:, :, :3]

        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        masked_img = img.copy()

        for i in range(masked_img.shape[0]):
            for j in range(masked_img.shape[1]):
                if meta[i, j, 0] != 0 or meta[i, j, 1] != 0:
                    masked_img[i, j, :3] = 0

                if meta[i, j, 0] != 0:
                    masked_img[i, j, 1] = 255
                if meta[i, j, 1] != 0:
                    masked_img[i, j, 2] = 255

        img = cv.resize(img, (800, 800))
        masked_img = cv.resize(masked_img, (800, 800))

        cv.imshow('masked', masked_img)
        cv.imshow('org', img)

        img = np.load(file)
        meta = img[:, :, 3:].copy()
        img = img.astype(np.uint8)[:, :, :3]

        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        masked_img = img.copy()
        for i in range(masked_img.shape[0]):
            for j in range(masked_img.shape[1]):

                if meta[i, j, 1] != 0:
                    masked_img[i, j, 0] = 0
                    masked_img[i, j, 1] = 0
                    masked_img[i, j, 2] = 255

        img = cv.resize(img, (800, 800))
        masked_img = cv.resize(masked_img, (1000, 1000))
        cv.imshow('segmentation', masked_img)

        cv.waitKey()