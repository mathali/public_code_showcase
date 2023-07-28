import cv2 as cv
import numpy as np
import glob
from pywt import dwt2


def process_contour_no_mirror(cX, cY, mask):
    start_y, end_y = max(cY - 7, 0), min(cY + 8, 255)
    start_x, end_x = max(cX - 7, 0), min(cX + 8, 255)

    len_y = end_y - start_y
    len_x = end_x - start_x

    mask[start_y:end_y, start_x:end_x] = 0

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

    patch = mask[start_y:end_y, start_x:end_x]
    if patch.shape != (15, 15, 3):
        assert 'Patch offset failed'

    return patch


def process_file(file, total_contrast, total_energy, total_roundness, total_eccentricity, total_MA, total_ma, total_extent, total_solidity,
                 r_mean, r_std, r_median, g_mean, g_std, g_median, b_mean, b_std, b_median):
    img = np.load(file)

    parent = file.split('=')[1].split('_c')[0].split('.npy')[0]
    y = file.split('=')[2].split('_')[0]
    x = file.split('=')[3].split('.')[0].split('_')[0]

    parent = np.load('TEST-DATA-PATH' + f'/{parent}.npy')
    mask = parent[:, :, 4].astype(np.uint8)
    patch_mask = process_contour_no_mirror(int(x), int(y), mask)
    contours, hierarchy = cv.findContours(patch_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    if len(contours) == 0 or np.where(patch_mask > 0, 1, 0).sum() < 20:
        for var in [total_energy, total_roundness, total_eccentricity, total_MA, total_ma, total_extent, total_solidity]:
            var += 0
    else:
        contour = max(contours, key=len)

        """ Calculate the profile roundness """
        # Measure of how close the contour is to a perfect circle
        moments = cv.moments(contour)
        length = cv.arcLength(contour, True)
        total_roundness += (length * length) / (moments['m00'] * 4 * np.pi)

        """ The eccentricity is calculated by using the fitted ellipse """
        # eccentricity ( Also called elongation ) It is a measure of the degree of contour elongation
        # Fit ellipse
        (x, y), (MA, ma), angle = cv.fitEllipse(contour)    # MA - major axis length; ma - minor axis length
        a = ma / 2
        b = MA / 2
        total_eccentricity += np.sqrt(a ** 2 - b ** 2) / a
        total_MA ++ MA
        total_ma += ma

        # Extent - Extent is the ratio of contour area to bounding rectangle area.
        area = cv.contourArea(contour)
        x, y, w, h = cv.boundingRect(contour)
        rect_area = w * h
        total_extent += float(area) / rect_area

        #Solidity - Solidity is the ratio of contour area to its convex hull area.
        area = cv.contourArea(contour)
        hull = cv.convexHull(contour)
        hull_area = cv.contourArea(hull)
        total_solidity += float(area) / hull_area

    # Image contrast
    img_grey = cv.cvtColor(img.transpose((1, 2, 0)).astype(np.uint8), cv.COLOR_RGB2GRAY)
    total_contrast += img_grey.std()

    # Image energy
    _, (cH, cV, cD) = dwt2(img_grey.T, 'db1')
    # a - LL, h - LH, v - HL, d - HH
    total_energy += (cH ** 2 + cV ** 2 + cD ** 2).sum() / img_grey.size

    # Colour metrics
    r_mean += img[0].mean()
    r_std += img[0].std()
    r_median += np.median(img[0])
    g_mean += img[1].mean()
    g_std += img[1].std()
    g_median += np.median(img[1])
    b_mean += img[2].mean()
    b_std += img[2].std()
    b_median += np.median(img[2])

    return total_contrast, total_energy, total_roundness, total_eccentricity, total_MA, total_ma, total_extent, total_solidity,\
           r_mean, r_std, r_median, g_mean, g_std, g_median, b_mean, b_std, b_median


def calculate_statistics():
    """
    DEPRECATED
    Calculates various image metrics as a first attempt at interpreting the model's behaviour
    The goal was to look at various metrics, class by class, and determine if there's a noticable pattern in the metrics
    when the model makes mistakes.

    Calculates: mean, median, std - for each channel
                contrast, energy, roundness, eccentricity, MA, extent, solidity - for the whole image
    """
    class_mappings = {0: 'Background', 1: 'Connective tissue', 2: 'Eosinophil', 3: 'Epithelial',
                      4: 'Lymhocyte', 5: 'Neutrophil', 6: 'Plasma'}

    for true in range(7):
        file_list = glob.glob(f'../outputs/test_statistics/file_mapping/{true}/{true}/*.npy')
        total_contrast, total_energy, total_roundness, total_eccentricity = 0, 0, 0, 0
        total_MA, total_ma, total_extent, total_solidity = 0, 0, 0, 0
        r_mean, r_std, r_median = 0, 0, 0
        g_mean, g_std, g_median = 0, 0, 0
        b_mean, b_std, b_median = 0, 0, 0
        for file in file_list:
            total_contrast, total_energy, total_roundness, total_eccentricity, total_MA, total_ma, total_extent, total_solidity,\
            r_mean, r_std, r_median, g_mean, g_std, g_median, b_mean, b_std, b_median = \
                process_file(file, total_contrast, total_energy, total_roundness, total_eccentricity, total_MA, total_ma, total_extent, total_solidity,
                             r_mean, r_std, r_median, g_mean, g_std, g_median, b_mean, b_std, b_median)

        print("=========================================================")
        print(f"Class: {class_mappings[true]}")
        print(f"True prediction descriptors:\tContrast:{total_contrast/len(file_list):.4f}\tEnergy:{total_energy/len(file_list):.4f}")
        print(f"\t\tRoundness:{total_roundness/len(file_list):.4f}\tEccentricity:{total_eccentricity/len(file_list):.4f}\tMajor axis length:{total_MA/len(file_list):.4f}\tMinor axis length:{total_ma/len(file_list):.4f}\tExtent:{total_extent/len(file_list):.4f}\tSolidity:{total_solidity/len(file_list):.4f}")
        print(f"R-channel:\tMean:{r_mean/len(file_list):.4f}\tSTD:{r_std/len(file_list):.4f}\tMedian:{r_median/len(file_list):.4f}")
        print(f"G-channel:\tMean:{r_mean/len(file_list):.4f}\tSTD:{r_std/len(file_list):.4f}\tMedian:{r_median/len(file_list):.4f}")
        print(f"B-channel:\tMean:{b_mean/len(file_list):.4f}\tSTD:{b_std/len(file_list):.4f}\tMedian:{b_median/len(file_list):.4f}")

        for pred in range(7):
            if pred == true:
                continue
            file_list = glob.glob(f'../outputs/test_statistics/file_mapping/{true}/{true}/*.npy')
            total_contrast, total_energy, total_roundness, total_eccentricity = 0, 0, 0, 0
            total_MA, total_ma, total_extent, total_solidity = 0, 0, 0, 0
            r_mean, r_std, r_median = 0, 0, 0
            g_mean, g_std, g_median = 0, 0, 0
            b_mean, b_std, b_median = 0, 0, 0
            for file in file_list:
                total_contrast, total_energy, total_roundness, total_eccentricity, total_MA, total_ma, total_extent, total_solidity,\
                r_mean, r_std, r_median, g_mean, g_std, g_median, b_mean, b_std, b_median = \
                    process_file(file, total_contrast, total_energy, total_roundness, total_eccentricity, total_MA, total_ma, total_extent, total_solidity,
                                 r_mean, r_std, r_median, g_mean, g_std, g_median, b_mean, b_std, b_median)

            print("------------------------------------------------------------")
            print(f"Class: {class_mappings[pred]}")
            print(f"False prediction descriptors:\tContrast:{total_contrast/len(file_list):.4f}\tEnergy:{total_energy/len(file_list):.4f}")
            print(f"\t\tRoundness:{total_roundness/len(file_list):.4f}\tEccentricity:{total_eccentricity/len(file_list):.4f}\tMajor axis length:{total_MA/len(file_list):.4f}\tMinor axis length:{total_ma/len(file_list):.4f}\tExtent:{total_extent/len(file_list):.4f}\tSolidity:{total_solidity/len(file_list):.4f}")
            print(f"R-channel:\tMean:{r_mean/len(file_list):.4f}\tSTD:{r_std/len(file_list):.4f}\tMedian:{r_median/len(file_list):.4f}")
            print(f"G-channel:\tMean:{r_mean/len(file_list):.4f}\tSTD:{r_std/len(file_list):.4f}\tMedian:{r_median/len(file_list):.4f}")
            print(f"B-channel:\tMean:{b_mean/len(file_list):.4f}\tSTD:{b_std/len(file_list):.4f}\tMedian:{b_median/len(file_list):.4f}")


if __name__ == '__main__':
    calculate_statistics()
