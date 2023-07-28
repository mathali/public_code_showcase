"""
# MIT License
# Copyright (c) 2017 Vooban Inc.
# Coded by: Guillaume Chevalier
# Source to original code and license:
#     https://github.com/Vooban/Smoothly-Blend-Image-Patches
#     https://github.com/Vooban/Smoothly-Blend-Image-Patches/blob/master/LICENSE

"""
"""Perform smooth predictions on an image from tiled prediction patches."""


import numpy as np
import scipy.signal
from tqdm import tqdm
import torch
import cv2

import gc


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    PLOT_PROGRESS = True
    # See end of file for the rest of the __main__.
else:
    import matplotlib.pyplot as plt
    PLOT_PROGRESS = False


def _spline_window(window_size, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """
    intersection = int(window_size/4)
    wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


cached_2d_windows = dict()
def _window_2D(window_size, power=2):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    # Memoization
    global cached_2d_windows
    key = "{}_{}".format(window_size, power)
    if key in cached_2d_windows:
        wind = cached_2d_windows[key]
    else:
        wind = _spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, 1), 1)      #SREENI: Changed from 3, 3, to 1, 1
        wind = wind * wind.transpose(1, 0, 2)
        if PLOT_PROGRESS:
            # For demo purpose, let's look once at the window:
            plt.imshow(wind[:, :, 0], cmap="viridis")
            plt.title("2D Windowing Function for a Smooth Blending of "
                      "Overlapping Patches")
            plt.show()
        cached_2d_windows[key] = wind
    return wind.transpose(2, 0, 1).astype(np.float32)


def _pad_img(img, window_size, subdivisions):
    """
    Add borders to img for a "valid" border pattern according to "window_size" and
    "subdivisions".
    Image is an np array of shape (time_steps, , nb_channels, x, y).
    """
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    more_borders = ((0, 0), (0, 0), (aug, aug), (aug, aug))
    ret = np.pad(img, pad_width=more_borders, mode='reflect')
    # gc.collect()

    if PLOT_PROGRESS:
        # For demo purpose, let's look once at the window:
        plt.imshow(np.transpose(np.squeeze(ret), (1, 2, 0)))
        plt.title("Padded Image for Using Tiled Prediction Patches\n"
                  "(notice the reflection effect on the padded borders)")
        plt.show()
    return ret


def _unpad_img(padded_img, window_size, subdivisions):
    """
    Undo what's done in the `_pad_img` function.
    Image is an np array of shape (x, y, nb_channels).
    """
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    ret = padded_img[
          :,
        aug:-aug,
        aug:-aug,
        # :
    ]
    # gc.collect()
    return ret


def _rotate_mirror_do(im):
    """
    Duplicate an np array (image) of shape (x, y, nb_channels) 8 times, in order
    to have all the possible rotations and mirrors of that image that fits the
    possible 90 degrees rotations.
    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    mirrs = []
    mirrs.append(np.array(im, dtype=np.float32))
    mirrs.append(np.rot90(np.array(im, dtype=np.float32), axes=(2, 3), k=1))
    mirrs.append(np.rot90(np.array(im, dtype=np.float32), axes=(2, 3), k=2))
    mirrs.append(np.rot90(np.array(im, dtype=np.float32), axes=(2, 3), k=3))
    return mirrs


def _rotate_mirror_undo(im_mirrs):
    """
    merges a list of 8 np arrays (images) of shape (x, y, nb_channels) generatedW
    from the `_rotate_mirror_do` function. Each images might have changed and
    merging them implies to rotated them back in order and average things out.
    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    origs = []
    origs.append(np.array(im_mirrs[0], dtype=np.float32))
    origs.append(np.rot90(np.array(im_mirrs[1], dtype=np.float32), axes=(1, 2), k=3))
    origs.append(np.rot90(np.array(im_mirrs[2], dtype=np.float32), axes=(1, 2), k=2))
    origs.append(np.rot90(np.array(im_mirrs[3], dtype=np.float32), axes=(1, 2), k=1))
    return np.mean(origs, axis=0)


def _windowed_subdivs(padded_img, window_size, subdivisions, nb_classes, pred_func, mode='multiclass_pyramid'):
    """
    Create tiled overlapping patches.
    Returns:
        5D numpy array of shape = (
            nb_patches_along_X,
            nb_patches_along_Y,
            patches_resolution_along_X,
            patches_resolution_along_Y,
            nb_output_channels
        )
    Note:
        patches_resolution_along_X == patches_resolution_along_Y == window_size
    """
    WINDOW_SPLINE_2D = _window_2D(window_size=window_size, power=2)

    step = int(window_size/subdivisions)
    padx_len = padded_img.shape[2]
    pady_len = padded_img.shape[3]
    subdivs = []

    for i in range(0, padx_len-window_size+1, step):
        subdivs.append([])
        for j in range(0, pady_len-window_size+1, step):
            if mode == 'multiclass_pyramid':
                tmp_downscaled = padded_img[:, :3, max(0, i - window_size//2):min(padded_img.shape[2], i + window_size + window_size//2),
                                                   max(0, j - window_size//2):min(padded_img.shape[3], j + window_size + window_size//2)]
                p = np.shape(tmp_downscaled)
                if p[2] != window_size*2 or p[3] != window_size * 2:
                    max_h = window_size * 2
                    max_w = window_size * 2
                    pad_h = (max_h - p[2])
                    pad_w = (max_w - p[3])

                    if pad_h <= 0:
                        pad_h_b = np.abs(pad_h)
                        pad_h_a = 0
                    else:
                        pad_h_b = 0
                        pad_h_a = pad_h

                    if pad_w <= 0:
                        pad_w_b = np.abs(pad_w)
                        pad_w_a = 0
                    else:
                        pad_w_b = 0
                        pad_w_a = pad_w

                    tmp_downscaled = cv2.copyMakeBorder(np.squeeze(tmp_downscaled).transpose(1, 2, 0),
                                                        pad_h_a, pad_h_b, pad_w_b, pad_w_a,
                                                        cv2.BORDER_REFLECT)
                else:
                    tmp_downscaled = np.squeeze(tmp_downscaled).transpose(1, 2, 0)

                tmp_downscaled = cv2.resize(tmp_downscaled, (window_size, window_size))

                tmp_downscaled = np.expand_dims(tmp_downscaled.transpose(2, 0, 1), axis=0)
                patch = np.zeros((padded_img.shape[0], padded_img.shape[1]+3, window_size, window_size))
                patch[:, :6] = padded_img[:, :, i:i+window_size, j:j+window_size]
                patch[:, 6:] = tmp_downscaled
            else:
                patch = padded_img[:, :, i:i+window_size, j:j+window_size]
            subdivs[-1].append(patch)

    # Here, `gc.collect()` clears RAM between operations.
    # It should run faster if they are removed, if enough memory is available.
    gc.collect()
    subdivs = np.array(subdivs)
    gc.collect()
    a, b, c, d, e, f = subdivs.shape
    if c != 1:
        c, d = d, c

    subdivs = subdivs.reshape(a * b, c, d, e, f)
    gc.collect()

    out_arr = np.empty((a*b, nb_classes, e, f), dtype=np.float32)
    device = torch.device("cuda:"+str(0)) if torch.cuda.is_available() else torch.device("cpu")
    for x in range(int(a*b/32)+1):
        batch = subdivs[x*32:(x+1)*32, :, :, :, :].astype(np.float32)
        batch = torch.from_numpy(np.squeeze(batch)).to(device)
        with torch.no_grad():
            if batch.any():
                out_arr[x * 32:(x + 1) * 32, :, :, :] = pred_func(batch).cpu().detach().numpy()

    # subdivs = pred_func(subdivs)
    gc.collect()
    out_arr = np.array([patch * WINDOW_SPLINE_2D for patch in out_arr], dtype=np.float32)
    # subdivs = np.array([patch * WINDOW_SPLINE_2D for patch in subdivs])
    # gc.collect()

    # Such 5D array:
    # subdivs = subdivs.reshape(a, b, c, d, e, nb_classes)
    out_arr = out_arr.reshape(a, b, nb_classes, e, f)
    # gc.collect()

    return out_arr


def _recreate_from_subdivs(subdivs, window_size, subdivisions, nb_classes, padded_out_shape):
    """
    Merge tiled overlapping patches smoothly.
    """
    step = int(window_size/subdivisions)
    padx_len = padded_out_shape[0]
    pady_len = padded_out_shape[1]

    y = np.zeros((nb_classes, padx_len, pady_len))

    a = 0
    # for i in range(0, padx_len-((subdivisions+1)*step), step):
    for i in range(0, padx_len-window_size+1, step):
        b = 0
    #    for j in range(0, pady_len-((subdivisions+1)*step), step):
        for j in range(0, pady_len-window_size+1, step):
            windowed_patch = subdivs[a, b]

            y[:, i:i+window_size, j:j+window_size] = y[:, i:i+window_size, j:j+window_size] + windowed_patch
            b += 1
        a += 1
    return y / (subdivisions ** 2)


def predict_img_with_smooth_windowing(input_img, window_size, subdivisions, nb_classes, pred_func, mode="multiclass_pyramid"):
    """
    Apply the `pred_func` function to square patches of the image, and overlap
    the predictions to merge them smoothly.
    See 6th, 7th and 8th idea here:
    http://blog.kaggle.com/2017/05/09/dstl-satellite-imagery-competition-3rd-place-winners-interview-vladimir-sergey/
    """
    pad = _pad_img(input_img, window_size, subdivisions)
    pads = _rotate_mirror_do(pad)

    # Note that the implementation could be more memory-efficient by merging
    # the behavior of `_windowed_subdivs` and `_recreate_from_subdivs` into
    # one loop doing in-place assignments to the new image matrix, rather than
    # using a temporary 5D array.

    # It would also be possible to allow different (and impure) window functions
    # that might not tile well. Adding their weighting to another matrix could
    # be done to later normalize the predictions correctly by dividing the whole
    # reconstructed thing by this matrix of weightings - to normalize things
    # back from an impure windowing function that would have badly weighted
    # windows.

    # For example, since the U-net of Kaggle's DSTL satellite imagery feature
    # prediction challenge's 3rd place winners use a different window size for
    # the input and output of the neural net's patches predictions, it would be
    # possible to fake a full-size window which would in fact just have a narrow
    # non-zero dommain. This may require to augment the `subdivisions` argument
    # to 4 rather than 2.

    res = []
    for pad in tqdm(pads):
        # For every rotation:
        sd = _windowed_subdivs(pad, window_size, subdivisions, nb_classes, pred_func, mode=mode)
        one_padded_result = _recreate_from_subdivs(
            sd, window_size, subdivisions, nb_classes=nb_classes,
            padded_out_shape=(pad.shape[-2], pad.shape[-1]))# list(pad.shape[:-1])+[nb_classes])

        res.append(one_padded_result)

    del pad, pads
    gc.collect()
    # Merge after rotations:
    padded_results = _rotate_mirror_undo(res)

    prd = _unpad_img(padded_results, window_size, subdivisions)

    # prd = prd[:input_img.shape[0], :input_img.shape[1], :]

    if PLOT_PROGRESS:
        plt.imshow(prd)
        plt.title("Smoothly Merged Patches that were Tiled Tighter")
        plt.show()
    return prd