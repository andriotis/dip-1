import numpy as np


def get_equalization_transform_of_img(img_array: np.ndarray) -> np.ndarray:

    H, W = img_array.shape
    L = img_array.max() - img_array.min() + 1

    levels = np.arange(L)
    occurences = np.zeros(L)

    for level in levels:
        occurences[level] = np.sum(img_array == level)
    pixel_probability = occurences / (H * W)

    v = np.array([np.sum(pixel_probability[: level + 1]) for level in levels])
    y = np.round(((v - v[0]) / (1 - v[0])) * (L - 1))

    equalization_transform = y.astype(np.uint8)

    return equalization_transform


def perform_global_hist_equalization(img_array: np.ndarray) -> np.ndarray:
    equalization_transform = get_equalization_transform_of_img(img_array)
    equalized_img = equalization_transform[img_array]
    return equalized_img
