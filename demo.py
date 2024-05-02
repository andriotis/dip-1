import numpy as np
import histogram.global_hist_eq as ghe
import histogram.adaptive_hist_eq as ahe
from histogram.utils import get_histogram
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load the input image and convert it to grayscale
    img_array = np.array(Image.open(fp="input_img.png").convert("L"))

    # Perform global histogram equalization
    img_global = ghe.perform_global_hist_equalization(img_array)

    # Perform adaptive histogram equalization
    img_local = ahe.perform_adaptive_hist_equalization(img_array, 64, 48)

    # Define the intensity levels
    levels = np.arange(256)

    # Plot the original image
    plt.figure(figsize=(10, 10))
    plt.imshow(
        img_array,
        cmap="gray",
    )
    plt.title("Original Image")

    # Plot the histogram of the original image
    plt.figure(figsize=(10, 10))
    plt.plot(
        levels,
        get_histogram(img_array),
    )
    plt.title("Histogram of Original Image")

    # Plot the global histogram equalized image
    plt.figure(figsize=(10, 10))
    plt.imshow(
        img_global,
        cmap="gray",
    )
    plt.title("Global Histogram Equalized Image")

    # Plot the histogram of the global histogram equalized image
    plt.figure(figsize=(10, 10))
    plt.plot(
        levels,
        get_histogram(img_global),
    )
    plt.title("Histogram of Global Histogram Equalized Image")

    # Plot the local histogram equalized image
    plt.figure(figsize=(10, 10))
    plt.imshow(
        img_local,
        cmap="gray",
    )
    plt.title("Local Histogram Equalized Image")

    # Plot the histogram of the local histogram equalized image
    plt.figure(figsize=(10, 10))
    plt.plot(
        levels,
        get_histogram(img_local),
    )
    plt.title("Histogram of Local Histogram Equalized Image")

    # Display the plots
    plt.show()
