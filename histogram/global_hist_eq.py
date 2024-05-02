import numpy as np


def get_equalization_transform_of_img(img_array: np.ndarray) -> np.ndarray:
    """
    Calculate the equalization transformation for an input image.

    Parameters:
    - img_array: Input image array (2D numpy array).

    Returns:
    - equalization_transform: Equalization transformation (1D numpy array).
    """
    # Get the height and width of the image
    H, W = img_array.shape
    # Define the number of intensity levels
    L = 256

    # Create an array of levels from 0 to L-1
    levels = np.arange(L)
    # Initialize an array to store the occurrences of each intensity level
    occurences = np.zeros(L)

    # Calculate the occurrences of each intensity level in the image
    for level in levels:
        occurences[level] = np.sum(img_array == level)

    # Calculate the probability of each intensity level
    pixel_probability = occurences / np.sum(occurences)

    # Calculate the cumulative distribution function (CDF)
    v = np.array([np.sum(pixel_probability[: level + 1]) for level in levels])

    # Perform histogram equalization transformation
    y = np.round(((v - v[0]) / (1 - v[0])) * (L - 1))

    # Convert the resulting transformation to uint8
    equalization_transform = y.astype(np.uint8)

    return equalization_transform


def perform_global_hist_equalization(img_array: np.ndarray) -> np.ndarray:
    """
    Perform global histogram equalization on the input image.

    Parameters:
    - img_array: Input image array (2D numpy array).

    Returns:
    - equalized_img: Equalized image array (2D numpy array).
    """
    # Calculate the equalization transformation
    equalization_transform = get_equalization_transform_of_img(img_array)

    # Apply the equalization transformation to the image
    equalized_img = equalization_transform[img_array]

    return equalized_img
