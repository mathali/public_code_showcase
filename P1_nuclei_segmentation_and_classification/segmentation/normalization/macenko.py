"""
Stain normalization based on the method of:

M. Macenko et al., ‘A method for normalizing histology slides for quantitative analysis’, in 2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro, 2009, pp. 1107–1110.

Uses the spams package:

http://spams-devel.gforge.inria.fr/index.html

Use with python via e.g https://anaconda.org/conda-forge/python-spams
"""

from __future__ import division

import numpy as np
from sklearn.decomposition import SparseCoder
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


def standardize_brightness(I):
    """

    :param I:
    :return:
    """
    p = np.percentile(I, 90)
    return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)


def remove_zeros(I):
    """
    Remove zeros, replace with 1's.
    :param I: uint8 array
    :return:
    """
    mask = (I == 0)
    I[mask] = 1
    return I


def RGB_to_OD(I):
    """
    Convert from RGB to optical density
    :param I:
    :return:
    """
    I = remove_zeros(I)
    return -1 * np.log(I / 255)


def OD_to_RGB(OD):
    """
    Convert from optical density to RGB
    :param OD:
    :return:
    """
    return (255 * np.exp(-1 * OD)).astype(np.uint8)


def normalize_rows(A):
    """
    Normalize rows of an array
    :param A:
    :return:
    """
    return A / np.linalg.norm(A, axis=1)[:, None]


def get_stain_matrix(I, beta=0.15, alpha=1):
    """
    Get stain matrix (2x3)
    :param I:
    :param beta:
    :param alpha:
    :return:
    """
    # 1. Convert to Optical Density
    OD = RGB_to_OD(I).reshape((-1, 3))

    # 2. Remove OD less than beta
    OD = (OD[(OD > beta).any(axis=1), :])

    # 3.1 Get eigenvectors
    _, V = np.linalg.eigh(np.cov(OD, rowvar=False))

    # 3.2 Make sure the vectors are ponting the right way
    V = V[:, [2, 1]]
    if V[0, 0] < 0: V[:, 0] *= -1
    if V[0, 1] < 0: V[:, 1] *= -1

    # 4. Project
    That = np.dot(OD, V)

    # 5. Calculate angle of each point in respect to the vector directions
    phi = np.arctan2(That[:, 1], That[:, 0])

    # 6. Find extremes
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    # 7. Min-max vectors coreesponding to Haematoxylin and Eosin
    v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))

    # 8. Order H fist, E second
    if v1[0] > v2[0]:
        HE = np.array([v1, v2])
    else:
        HE = np.array([v2, v1])

    return normalize_rows(HE)


def get_concentrations(I, stain_matrix, lamda=0.01):
    """
    Get concentrations, a npix x 2 matrix
    :param I:
    :param stain_matrix: a 2x3 stain matrix
    :return:
    """
    OD = RGB_to_OD(I).reshape((-1, 3))
    coder = SparseCoder(dictionary=stain_matrix,  transform_algorithm='lasso_lars', positive_code=True, transform_alpha=lamda)
    return coder.transform(OD)
    # return spams.lasso(OD.T, D=stain_matrix.T, mode=2, lambda1=lamda, pos=True).toarray().T

###
class Normalizer(object):
    """
    A stain normalization object
    """

    def __init__(self):
        self.stain_matrix_target = None
        self.target_concentrations = None

    def fit(self, target):
        target = standardize_brightness(target)
        # Steps 1. - 8. of the Macenko algorithm
        self.stain_matrix_target = get_stain_matrix(target)

        # 9. Determine concentrations of individual stains
        self.target_concentrations = get_concentrations(target, self.stain_matrix_target)

    def target_stains(self):
        return OD_to_RGB(self.stain_matrix_target)

    def transform(self, I):
        I = standardize_brightness(I)
        stain_matrix_source = get_stain_matrix(I)
        source_concentrations = get_concentrations(I, stain_matrix_source)
        maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        maxC_target = np.percentile(self.target_concentrations, 99, axis=0).reshape((1, 2))
        source_concentrations *= (maxC_target / maxC_source)
        return (255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target).reshape(I.shape))).astype(
            np.uint8)

    def hematoxylin(self, I):
        I = standardize_brightness(I)
        h, w, c = I.shape
        stain_matrix_source = get_stain_matrix(I)
        source_concentrations = get_concentrations(I, stain_matrix_source)
        H = source_concentrations[:, 0].reshape(h, w)
        H = np.exp(-1 * H)
        return H


def process_file(file, normalizer, mode):
    target = np.load(file)
    rgb = target[:, :, :3]
    meta = target[:, :, 3:]

    name = file.split('\\')[-1]
    type = name.split('_')[0]

    try:
        normalized = normalizer.transform(rgb)

        out = np.transpose(normalized, (2, 0, 1))
        meta = np.transpose(meta, (2, 0, 1))
        out = np.vstack((out, meta))
        out = np.transpose(out, (1, 2, 0))

        np.save("OUTPUT-PATH" + f'/{name}', out)
    except np.linalg.LinAlgError as e:
        pass

    return 1

# Perform Macenko's normalization method on the dataset
# Don't forget to specify paths to:
# 1. The target image (normalization baseline)
# 2. Input dataset
# 3. Output directory
if __name__ == '__main__':
    img = np.load("TARGET-IMAGE-PATH")
    rgb_img = img[:, :, :3]
    normalizer = Normalizer()
    normalizer.fit(rgb_img)

    for mode in ['valid', 'test']:
        file_list = glob.glob("DATSET-PATH" + '/*.npy')
        print(f'Processing mode: {mode}')
        norm_func = partial(process_file, normalizer=normalizer, mode=mode)

        with Pool(12) as p:
            list(tqdm(p.imap(norm_func, file_list), total=len(file_list)))
