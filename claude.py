import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from PIL import Image
from typing import Dict, Tuple


def get_equalization_transform_of_img(img_array: np.ndarray) -> np.ndarray:

    H, W = img_array.shape
    L = 256

    levels = np.arange(L)
    occurences = np.zeros(L)

    for level in levels:
        occurences[level] = np.sum(img_array == level)
    pixel_probability = occurences / np.sum(occurences)

    v = np.array([np.sum(pixel_probability[: level + 1]) for level in levels])

    y = np.round(((v - v[0]) / (1 - v[0])) * (L - 1))

    equalization_transform = y.astype(np.uint8)

    return equalization_transform


def perform_global_hist_equalization(img_array: np.ndarray) -> np.ndarray:
    equalization_transform = get_equalization_transform_of_img(img_array)
    equalized_img = equalization_transform[img_array]
    return equalized_img


def calculate_eq_transformations_of_regions(
    img_array: np.ndarray, region_len_h: int, region_len_w: int
) -> Dict[Tuple[int, int], np.ndarray]:

    H, W = img_array.shape
    num_regions_H = H // region_len_h
    num_regions_W = W // region_len_w

    region_to_eq_transform = {}

    centres = [
        (x, y)
        for x in np.linspace(
            0 + region_len_h // 2, H - region_len_h // 2, num_regions_H, dtype=int
        )
        for y in np.linspace(
            0 + region_len_w // 2, W - region_len_w // 2, num_regions_W, dtype=int
        )
    ]

    for centre_h, centre_w in centres:

        region_to_eq_transform[(centre_h, centre_w)] = (
            get_equalization_transform_of_img(
                img_array[
                    centre_h - region_len_h // 2 : centre_h + region_len_h // 2,
                    centre_w - region_len_w // 2 : centre_w + region_len_w // 2,
                ]
            )
        )

    return region_to_eq_transform


def perform_adaptive_hist_equalization(
    img_array: np.ndarray, region_len_h: int, region_len_w: int
) -> np.ndarray:

    copy_img_array = img_array.copy()

    H, W = copy_img_array.shape

    num_regions_H = H // region_len_h
    num_regions_W = W // region_len_w

    new_H = num_regions_H * region_len_h
    new_W = num_regions_W * region_len_w

    cropped_img_array = copy_img_array[
        (H - new_H) // 2 : (H - new_H) // 2 + new_H,
        (W - new_W) // 2 : (W - new_W) // 2 + new_W,
    ]

    equalized_img = np.zeros((new_H, new_W))

    region_to_eq_transform = calculate_eq_transformations_of_regions(
        cropped_img_array, region_len_h, region_len_w
    )

    for index, ((centre_h, centre_w), T) in enumerate(region_to_eq_transform.items()):
        equalized_img[(centre_h, centre_w)] = 255
        # top-left region
        if index == 0:
            for j in np.arange(
                centre_w - region_len_w // 2, centre_w + region_len_w // 2
            ):
                for i in np.arange(centre_h - region_len_h // 2, centre_h):
                    equalized_img[i, j] = T[cropped_img_array[i, j]]
                for i in np.arange(centre_h, centre_h + region_len_h // 2):
                    equalized_img[i, j] = T[cropped_img_array[i, j]]
                for i in np.arange(centre_h, centre_h + region_len_h // 2):
                    # bottom-right quadrant
                    upper_left = (centre_h, centre_w)
                    upper_right = (centre_h, centre_w + region_len_w)
                    bottom_left = (centre_h + region_len_h, centre_w)
                    bottom_right = (centre_h + region_len_h, centre_w + region_len_w)
                    for j in np.arange(centre_w, centre_w + region_len_w // 2):
                        R1 = region_to_eq_transform[upper_left][
                            cropped_img_array[i, j]
                        ] * (
                            (upper_right[1] - j) / (upper_right[1] - upper_left[1])
                        ) + region_to_eq_transform[
                            upper_right
                        ][
                            cropped_img_array[i, j]
                        ] * (
                            (j - upper_left[1]) / (upper_right[1] - upper_left[1])
                        )
                        R2 = region_to_eq_transform[bottom_left][
                            cropped_img_array[i, j]
                        ] * (
                            (bottom_right[1] - j) / (bottom_right[1] - bottom_left[1])
                        ) + region_to_eq_transform[
                            bottom_right
                        ][
                            cropped_img_array[i, j]
                        ] * (
                            (j - bottom_left[1]) / (bottom_right[1] - bottom_left[1])
                        )

                        equalized_img[i, j] = R1 * (
                            (bottom_left[0] - i) / (bottom_left[0] - upper_left[0])
                        ) + R2 * (
                            (i - upper_left[0]) / (bottom_left[0] - upper_left[0])
                        )
        # top-right region
        if index == num_regions_W - 1:
            for j in np.arange(
                centre_w - region_len_w // 2, centre_w + region_len_w // 2
            ):
                for i in np.arange(centre_h - region_len_h // 2, centre_h):
                    equalized_img[i, j] = T[cropped_img_array[i, j]]
            for j in np.arange(centre_w, centre_w + region_len_w // 2):
                for i in np.arange(centre_h, centre_h + region_len_h // 2):
                    equalized_img[i, j] = T[cropped_img_array[i, j]]

            # for points in bottom quadrant of region
            for i in np.arange(centre_h, centre_h + region_len_h // 2):

                for j in np.arange(centre_w - region_len_w // 2, centre_w):

                    # bottom-left quadrant
                    upper_left = (centre_h, centre_w - region_len_w)
                    upper_right = (centre_h, centre_w)
                    bottom_left = (centre_h + region_len_h, centre_w - region_len_w)
                    bottom_right = (centre_h + region_len_h, centre_w)

                    R1 = region_to_eq_transform[upper_left][cropped_img_array[i, j]] * (
                        (upper_right[1] - j) / (upper_right[1] - upper_left[1])
                    ) + region_to_eq_transform[upper_right][cropped_img_array[i, j]] * (
                        (j - upper_left[1]) / (upper_right[1] - upper_left[1])
                    )
                    R2 = region_to_eq_transform[bottom_left][
                        cropped_img_array[i, j]
                    ] * (
                        (bottom_right[1] - j) / (bottom_right[1] - bottom_left[1])
                    ) + region_to_eq_transform[
                        bottom_right
                    ][
                        cropped_img_array[i, j]
                    ] * (
                        (j - bottom_left[1]) / (bottom_right[1] - bottom_left[1])
                    )

                    equalized_img[i, j] = R1 * (
                        (bottom_left[0] - i) / (bottom_left[0] - upper_left[0])
                    ) + R2 * ((i - upper_left[0]) / (bottom_left[0] - upper_left[0]))

        # bottom-left region
        if index == (num_regions_H - 1) * num_regions_W:
            for i in np.arange(centre_h - region_len_h // 2, centre_h):
                for j in np.arange(centre_w - region_len_w // 2, centre_w):
                    equalized_img[i, j] = T[cropped_img_array[i, j]]
                for j in np.arange(centre_w, centre_w + region_len_w // 2):

                    # top-right quadrant
                    upper_left = (centre_h - region_len_h, centre_w)
                    upper_right = (centre_h - region_len_h, centre_w + region_len_w)
                    bottom_left = (centre_h, centre_w)
                    bottom_right = (centre_h, centre_w + region_len_w)

                    R1 = region_to_eq_transform[upper_left][cropped_img_array[i, j]] * (
                        (upper_right[1] - j) / (upper_right[1] - upper_left[1])
                    ) + region_to_eq_transform[upper_right][cropped_img_array[i, j]] * (
                        (j - upper_left[1]) / (upper_right[1] - upper_left[1])
                    )
                    R2 = region_to_eq_transform[bottom_left][
                        cropped_img_array[i, j]
                    ] * (
                        (bottom_right[1] - j) / (bottom_right[1] - bottom_left[1])
                    ) + region_to_eq_transform[
                        bottom_right
                    ][
                        cropped_img_array[i, j]
                    ] * (
                        (j - bottom_left[1]) / (bottom_right[1] - bottom_left[1])
                    )

                    equalized_img[i, j] = R1 * (
                        (bottom_left[0] - i) / (bottom_left[0] - upper_left[0])
                    ) + R2 * ((i - upper_left[0]) / (bottom_left[0] - upper_left[0]))

            for i in np.arange(centre_h, centre_h + region_len_h // 2):
                for j in np.arange(
                    centre_w - region_len_w // 2, centre_w + region_len_w // 2
                ):
                    equalized_img[i, j] = T[cropped_img_array[i, j]]

        # bottom-right region
        if index == num_regions_W * num_regions_H - 1:
            for i in np.arange(centre_h - region_len_h // 2, centre_h):
                for j in np.arange(centre_w - region_len_w // 2, centre_w):

                    # top-left quadrant
                    upper_left = (centre_h - region_len_h, centre_w - region_len_w)
                    upper_right = (centre_h - region_len_h, centre_w)
                    bottom_left = (centre_h, centre_w - region_len_w)
                    bottom_right = (centre_h, centre_w)

                    R1 = region_to_eq_transform[upper_left][cropped_img_array[i, j]] * (
                        (upper_right[1] - j) / (upper_right[1] - upper_left[1])
                    ) + region_to_eq_transform[upper_right][cropped_img_array[i, j]] * (
                        (j - upper_left[1]) / (upper_right[1] - upper_left[1])
                    )
                    R2 = region_to_eq_transform[bottom_left][
                        cropped_img_array[i, j]
                    ] * (
                        (bottom_right[1] - j) / (bottom_right[1] - bottom_left[1])
                    ) + region_to_eq_transform[
                        bottom_right
                    ][
                        cropped_img_array[i, j]
                    ] * (
                        (j - bottom_left[1]) / (bottom_right[1] - bottom_left[1])
                    )

                    equalized_img[i, j] = R1 * (
                        (bottom_left[0] - i) / (bottom_left[0] - upper_left[0])
                    ) + R2 * ((i - upper_left[0]) / (bottom_left[0] - upper_left[0]))

                for j in np.arange(centre_w, centre_w + region_len_w // 2):
                    equalized_img[i, j] = T[cropped_img_array[i, j]]

            for i in np.arange(centre_h, centre_h + region_len_h // 2):
                for j in np.arange(
                    centre_w - region_len_w // 2, centre_w + region_len_w // 2
                ):
                    equalized_img[i, j] = T[cropped_img_array[i, j]]

        if index > 0 and index < num_regions_W - 1:
            for i in np.arange(centre_h - region_len_h // 2, centre_h):
                for j in np.arange(
                    centre_w - region_len_w // 2, centre_w + region_len_w // 2
                ):
                    equalized_img[i, j] = T[cropped_img_array[i, j]]
            # for points in bottom quadrant of region
            for i in np.arange(centre_h, centre_h + region_len_h // 2):

                for j in np.arange(centre_w - region_len_w // 2, centre_w):

                    # bottom-left quadrant
                    upper_left = (centre_h, centre_w - region_len_w)
                    upper_right = (centre_h, centre_w)
                    bottom_left = (centre_h + region_len_h, centre_w - region_len_w)
                    bottom_right = (centre_h + region_len_h, centre_w)

                    R1 = region_to_eq_transform[upper_left][cropped_img_array[i, j]] * (
                        (upper_right[1] - j) / (upper_right[1] - upper_left[1])
                    ) + region_to_eq_transform[upper_right][cropped_img_array[i, j]] * (
                        (j - upper_left[1]) / (upper_right[1] - upper_left[1])
                    )
                    R2 = region_to_eq_transform[bottom_left][
                        cropped_img_array[i, j]
                    ] * (
                        (bottom_right[1] - j) / (bottom_right[1] - bottom_left[1])
                    ) + region_to_eq_transform[
                        bottom_right
                    ][
                        cropped_img_array[i, j]
                    ] * (
                        (j - bottom_left[1]) / (bottom_right[1] - bottom_left[1])
                    )

                    equalized_img[i, j] = R1 * (
                        (bottom_left[0] - i) / (bottom_left[0] - upper_left[0])
                    ) + R2 * ((i - upper_left[0]) / (bottom_left[0] - upper_left[0]))

                for j in np.arange(centre_w, centre_w + region_len_w // 2):

                    # bottom-right quadrant
                    upper_left = (centre_h, centre_w)
                    upper_right = (centre_h, centre_w + region_len_w)
                    bottom_left = (centre_h + region_len_h, centre_w)
                    bottom_right = (centre_h + region_len_h, centre_w + region_len_w)

                    R1 = region_to_eq_transform[upper_left][cropped_img_array[i, j]] * (
                        (upper_right[1] - j) / (upper_right[1] - upper_left[1])
                    ) + region_to_eq_transform[upper_right][cropped_img_array[i, j]] * (
                        (j - upper_left[1]) / (upper_right[1] - upper_left[1])
                    )
                    R2 = region_to_eq_transform[bottom_left][
                        cropped_img_array[i, j]
                    ] * (
                        (bottom_right[1] - j) / (bottom_right[1] - bottom_left[1])
                    ) + region_to_eq_transform[
                        bottom_right
                    ][
                        cropped_img_array[i, j]
                    ] * (
                        (j - bottom_left[1]) / (bottom_right[1] - bottom_left[1])
                    )

                    equalized_img[i, j] = R1 * (
                        (bottom_left[0] - i) / (bottom_left[0] - upper_left[0])
                    ) + R2 * ((i - upper_left[0]) / (bottom_left[0] - upper_left[0]))

        if (
            index > (num_regions_H - 1) * num_regions_W
            and index < num_regions_W * num_regions_H - 1
        ):
            for i in np.arange(centre_h, centre_h + region_len_h // 2):
                for j in np.arange(
                    centre_w - region_len_w // 2, centre_w + region_len_w // 2
                ):
                    equalized_img[i, j] = T[cropped_img_array[i, j]]

            # for points in top quadrant of region
            for i in np.arange(centre_h - region_len_h // 2, centre_h):

                for j in np.arange(centre_w - region_len_w // 2, centre_w):

                    # top-left quadrant
                    upper_left = (centre_h - region_len_h, centre_w - region_len_w)
                    upper_right = (centre_h - region_len_h, centre_w)
                    bottom_left = (centre_h, centre_w - region_len_w)
                    bottom_right = (centre_h, centre_w)

                    R1 = region_to_eq_transform[upper_left][cropped_img_array[i, j]] * (
                        (upper_right[1] - j) / (upper_right[1] - upper_left[1])
                    ) + region_to_eq_transform[upper_right][cropped_img_array[i, j]] * (
                        (j - upper_left[1]) / (upper_right[1] - upper_left[1])
                    )
                    R2 = region_to_eq_transform[bottom_left][
                        cropped_img_array[i, j]
                    ] * (
                        (bottom_right[1] - j) / (bottom_right[1] - bottom_left[1])
                    ) + region_to_eq_transform[
                        bottom_right
                    ][
                        cropped_img_array[i, j]
                    ] * (
                        (j - bottom_left[1]) / (bottom_right[1] - bottom_left[1])
                    )

                    equalized_img[i, j] = R1 * (
                        (bottom_left[0] - i) / (bottom_left[0] - upper_left[0])
                    ) + R2 * ((i - upper_left[0]) / (bottom_left[0] - upper_left[0]))

                for j in np.arange(centre_w, centre_w + region_len_w // 2):

                    # top-right quadrant
                    upper_left = (centre_h - region_len_h, centre_w)
                    upper_right = (centre_h - region_len_h, centre_w + region_len_w)
                    bottom_left = (centre_h, centre_w)
                    bottom_right = (centre_h, centre_w + region_len_w)

                    R1 = region_to_eq_transform[upper_left][cropped_img_array[i, j]] * (
                        (upper_right[1] - j) / (upper_right[1] - upper_left[1])
                    ) + region_to_eq_transform[upper_right][cropped_img_array[i, j]] * (
                        (j - upper_left[1]) / (upper_right[1] - upper_left[1])
                    )
                    R2 = region_to_eq_transform[bottom_left][
                        cropped_img_array[i, j]
                    ] * (
                        (bottom_right[1] - j) / (bottom_right[1] - bottom_left[1])
                    ) + region_to_eq_transform[
                        bottom_right
                    ][
                        cropped_img_array[i, j]
                    ] * (
                        (j - bottom_left[1]) / (bottom_right[1] - bottom_left[1])
                    )

                    equalized_img[i, j] = R1 * (
                        (bottom_left[0] - i) / (bottom_left[0] - upper_left[0])
                    ) + R2 * ((i - upper_left[0]) / (bottom_left[0] - upper_left[0]))

        if (
            index % num_regions_W == 0
            and index != 0
            and index != (num_regions_H - 1) * num_regions_W
        ):
            for j in np.arange(centre_w - region_len_w // 2, centre_w):
                for i in np.arange(
                    centre_h - region_len_h // 2, centre_h + region_len_h // 2
                ):
                    equalized_img[i, j] = T[cropped_img_array[i, j]]
            for i in np.arange(centre_h - region_len_h // 2, centre_h):
                for j in np.arange(centre_w, centre_w + region_len_w // 2):

                    # top-right quadrant
                    upper_left = (centre_h - region_len_h, centre_w)
                    upper_right = (centre_h - region_len_h, centre_w + region_len_w)
                    bottom_left = (centre_h, centre_w)
                    bottom_right = (centre_h, centre_w + region_len_w)

                    R1 = region_to_eq_transform[upper_left][cropped_img_array[i, j]] * (
                        (upper_right[1] - j) / (upper_right[1] - upper_left[1])
                    ) + region_to_eq_transform[upper_right][cropped_img_array[i, j]] * (
                        (j - upper_left[1]) / (upper_right[1] - upper_left[1])
                    )
                    R2 = region_to_eq_transform[bottom_left][
                        cropped_img_array[i, j]
                    ] * (
                        (bottom_right[1] - j) / (bottom_right[1] - bottom_left[1])
                    ) + region_to_eq_transform[
                        bottom_right
                    ][
                        cropped_img_array[i, j]
                    ] * (
                        (j - bottom_left[1]) / (bottom_right[1] - bottom_left[1])
                    )

                    equalized_img[i, j] = R1 * (
                        (bottom_left[0] - i) / (bottom_left[0] - upper_left[0])
                    ) + R2 * ((i - upper_left[0]) / (bottom_left[0] - upper_left[0]))

            for i in np.arange(centre_h, centre_h + region_len_h // 2):
                for j in np.arange(centre_w, centre_w + region_len_w // 2):

                    # bottom-right quadrant
                    upper_left = (centre_h, centre_w)
                    upper_right = (centre_h, centre_w + region_len_w)
                    bottom_left = (centre_h + region_len_h, centre_w)
                    bottom_right = (centre_h + region_len_h, centre_w + region_len_w)

                    R1 = region_to_eq_transform[upper_left][cropped_img_array[i, j]] * (
                        (upper_right[1] - j) / (upper_right[1] - upper_left[1])
                    ) + region_to_eq_transform[upper_right][cropped_img_array[i, j]] * (
                        (j - upper_left[1]) / (upper_right[1] - upper_left[1])
                    )
                    R2 = region_to_eq_transform[bottom_left][
                        cropped_img_array[i, j]
                    ] * (
                        (bottom_right[1] - j) / (bottom_right[1] - bottom_left[1])
                    ) + region_to_eq_transform[
                        bottom_right
                    ][
                        cropped_img_array[i, j]
                    ] * (
                        (j - bottom_left[1]) / (bottom_right[1] - bottom_left[1])
                    )

                    equalized_img[i, j] = R1 * (
                        (bottom_left[0] - i) / (bottom_left[0] - upper_left[0])
                    ) + R2 * ((i - upper_left[0]) / (bottom_left[0] - upper_left[0]))

        if (
            index % num_regions_W == num_regions_W - 1
            and index > num_regions_W - 1
            and index < num_regions_H * num_regions_W - 1
        ):
            for j in np.arange(centre_w, centre_w + region_len_w // 2):
                for i in np.arange(
                    centre_h - region_len_h // 2, centre_h + region_len_h // 2
                ):
                    equalized_img[i, j] = T[cropped_img_array[i, j]]

            # for points in top quadrant of region
            for i in np.arange(centre_h - region_len_h // 2, centre_h):

                for j in np.arange(centre_w - region_len_w // 2, centre_w):

                    # top-left quadrant
                    upper_left = (centre_h - region_len_h, centre_w - region_len_w)
                    upper_right = (centre_h - region_len_h, centre_w)
                    bottom_left = (centre_h, centre_w - region_len_w)
                    bottom_right = (centre_h, centre_w)

                    R1 = region_to_eq_transform[upper_left][cropped_img_array[i, j]] * (
                        (upper_right[1] - j) / (upper_right[1] - upper_left[1])
                    ) + region_to_eq_transform[upper_right][cropped_img_array[i, j]] * (
                        (j - upper_left[1]) / (upper_right[1] - upper_left[1])
                    )
                    R2 = region_to_eq_transform[bottom_left][
                        cropped_img_array[i, j]
                    ] * (
                        (bottom_right[1] - j) / (bottom_right[1] - bottom_left[1])
                    ) + region_to_eq_transform[
                        bottom_right
                    ][
                        cropped_img_array[i, j]
                    ] * (
                        (j - bottom_left[1]) / (bottom_right[1] - bottom_left[1])
                    )

                    equalized_img[i, j] = R1 * (
                        (bottom_left[0] - i) / (bottom_left[0] - upper_left[0])
                    ) + R2 * ((i - upper_left[0]) / (bottom_left[0] - upper_left[0]))

            # for points in bottom quadrant of region
            for i in np.arange(centre_h, centre_h + region_len_h // 2):

                for j in np.arange(centre_w - region_len_w // 2, centre_w):

                    # bottom-left quadrant
                    upper_left = (centre_h, centre_w - region_len_w)
                    upper_right = (centre_h, centre_w)
                    bottom_left = (centre_h + region_len_h, centre_w - region_len_w)
                    bottom_right = (centre_h + region_len_h, centre_w)

                    R1 = region_to_eq_transform[upper_left][cropped_img_array[i, j]] * (
                        (upper_right[1] - j) / (upper_right[1] - upper_left[1])
                    ) + region_to_eq_transform[upper_right][cropped_img_array[i, j]] * (
                        (j - upper_left[1]) / (upper_right[1] - upper_left[1])
                    )
                    R2 = region_to_eq_transform[bottom_left][
                        cropped_img_array[i, j]
                    ] * (
                        (bottom_right[1] - j) / (bottom_right[1] - bottom_left[1])
                    ) + region_to_eq_transform[
                        bottom_right
                    ][
                        cropped_img_array[i, j]
                    ] * (
                        (j - bottom_left[1]) / (bottom_right[1] - bottom_left[1])
                    )

                    equalized_img[i, j] = R1 * (
                        (bottom_left[0] - i) / (bottom_left[0] - upper_left[0])
                    ) + R2 * ((i - upper_left[0]) / (bottom_left[0] - upper_left[0]))

        if (centre_h > region_len_h // 2 and centre_h < new_H - region_len_h // 2) and (
            centre_w > region_len_w // 2 and centre_w < new_W - region_len_w // 2
        ):

            # for points in top quadrant of region
            for i in np.arange(centre_h - region_len_h // 2, centre_h):

                for j in np.arange(centre_w - region_len_w // 2, centre_w):

                    # top-left quadrant
                    upper_left = (centre_h - region_len_h, centre_w - region_len_w)
                    upper_right = (centre_h - region_len_h, centre_w)
                    bottom_left = (centre_h, centre_w - region_len_w)
                    bottom_right = (centre_h, centre_w)

                    R1 = region_to_eq_transform[upper_left][cropped_img_array[i, j]] * (
                        (upper_right[1] - j) / (upper_right[1] - upper_left[1])
                    ) + region_to_eq_transform[upper_right][cropped_img_array[i, j]] * (
                        (j - upper_left[1]) / (upper_right[1] - upper_left[1])
                    )
                    R2 = region_to_eq_transform[bottom_left][
                        cropped_img_array[i, j]
                    ] * (
                        (bottom_right[1] - j) / (bottom_right[1] - bottom_left[1])
                    ) + region_to_eq_transform[
                        bottom_right
                    ][
                        cropped_img_array[i, j]
                    ] * (
                        (j - bottom_left[1]) / (bottom_right[1] - bottom_left[1])
                    )

                    equalized_img[i, j] = R1 * (
                        (bottom_left[0] - i) / (bottom_left[0] - upper_left[0])
                    ) + R2 * ((i - upper_left[0]) / (bottom_left[0] - upper_left[0]))

                for j in np.arange(centre_w, centre_w + region_len_w // 2):

                    # top-right quadrant
                    upper_left = (centre_h - region_len_h, centre_w)
                    upper_right = (centre_h - region_len_h, centre_w + region_len_w)
                    bottom_left = (centre_h, centre_w)
                    bottom_right = (centre_h, centre_w + region_len_w)

                    R1 = region_to_eq_transform[upper_left][cropped_img_array[i, j]] * (
                        (upper_right[1] - j) / (upper_right[1] - upper_left[1])
                    ) + region_to_eq_transform[upper_right][cropped_img_array[i, j]] * (
                        (j - upper_left[1]) / (upper_right[1] - upper_left[1])
                    )
                    R2 = region_to_eq_transform[bottom_left][
                        cropped_img_array[i, j]
                    ] * (
                        (bottom_right[1] - j) / (bottom_right[1] - bottom_left[1])
                    ) + region_to_eq_transform[
                        bottom_right
                    ][
                        cropped_img_array[i, j]
                    ] * (
                        (j - bottom_left[1]) / (bottom_right[1] - bottom_left[1])
                    )

                    equalized_img[i, j] = R1 * (
                        (bottom_left[0] - i) / (bottom_left[0] - upper_left[0])
                    ) + R2 * ((i - upper_left[0]) / (bottom_left[0] - upper_left[0]))

            # for points in bottom quadrant of region
            for i in np.arange(centre_h, centre_h + region_len_h // 2):

                for j in np.arange(centre_w - region_len_w // 2, centre_w):

                    # bottom-left quadrant
                    upper_left = (centre_h, centre_w - region_len_w)
                    upper_right = (centre_h, centre_w)
                    bottom_left = (centre_h + region_len_h, centre_w - region_len_w)
                    bottom_right = (centre_h + region_len_h, centre_w)

                    R1 = region_to_eq_transform[upper_left][cropped_img_array[i, j]] * (
                        (upper_right[1] - j) / (upper_right[1] - upper_left[1])
                    ) + region_to_eq_transform[upper_right][cropped_img_array[i, j]] * (
                        (j - upper_left[1]) / (upper_right[1] - upper_left[1])
                    )
                    R2 = region_to_eq_transform[bottom_left][
                        cropped_img_array[i, j]
                    ] * (
                        (bottom_right[1] - j) / (bottom_right[1] - bottom_left[1])
                    ) + region_to_eq_transform[
                        bottom_right
                    ][
                        cropped_img_array[i, j]
                    ] * (
                        (j - bottom_left[1]) / (bottom_right[1] - bottom_left[1])
                    )

                    equalized_img[i, j] = R1 * (
                        (bottom_left[0] - i) / (bottom_left[0] - upper_left[0])
                    ) + R2 * ((i - upper_left[0]) / (bottom_left[0] - upper_left[0]))

                for j in np.arange(centre_w, centre_w + region_len_w // 2):

                    # bottom-right quadrant
                    upper_left = (centre_h, centre_w)
                    upper_right = (centre_h, centre_w + region_len_w)
                    bottom_left = (centre_h + region_len_h, centre_w)
                    bottom_right = (centre_h + region_len_h, centre_w + region_len_w)

                    R1 = region_to_eq_transform[upper_left][cropped_img_array[i, j]] * (
                        (upper_right[1] - j) / (upper_right[1] - upper_left[1])
                    ) + region_to_eq_transform[upper_right][cropped_img_array[i, j]] * (
                        (j - upper_left[1]) / (upper_right[1] - upper_left[1])
                    )
                    R2 = region_to_eq_transform[bottom_left][
                        cropped_img_array[i, j]
                    ] * (
                        (bottom_right[1] - j) / (bottom_right[1] - bottom_left[1])
                    ) + region_to_eq_transform[
                        bottom_right
                    ][
                        cropped_img_array[i, j]
                    ] * (
                        (j - bottom_left[1]) / (bottom_right[1] - bottom_left[1])
                    )

                    equalized_img[i, j] = R1 * (
                        (bottom_left[0] - i) / (bottom_left[0] - upper_left[0])
                    ) + R2 * ((i - upper_left[0]) / (bottom_left[0] - upper_left[0]))

    return equalized_img


import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from skimage import data
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk
from skimage.morphology import ball
from skimage.filters import rank


matplotlib.rcParams["font.size"] = 9


def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram."""
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap="gray")
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins)
    ax_hist.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax_hist.set_xlabel("Pixel intensity")

    xmin, xmax = dtype_range[image.dtype.type]
    ax_hist.set_xlim(xmin, xmax)

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, "r")

    return ax_img, ax_hist, ax_cdf


if __name__ == "__main__":

    img_array = np.array(Image.open(fp="input_img.png").convert("L"))
    # Global equalize
    img_rescale = exposure.equalize_hist(img_array)

    # Equalization
    footprint = disk(50)
    img_eq = rank.equalize(img_array, footprint=footprint)

    # Display results
    fig = plt.figure(figsize=(8, 5))
    axes = np.zeros((2, 3), dtype=object)
    axes[0, 0] = plt.subplot(2, 3, 1)
    axes[0, 1] = plt.subplot(2, 3, 2, sharex=axes[0, 0], sharey=axes[0, 0])
    axes[0, 2] = plt.subplot(2, 3, 3, sharex=axes[0, 0], sharey=axes[0, 0])
    axes[1, 0] = plt.subplot(2, 3, 4)
    axes[1, 1] = plt.subplot(2, 3, 5)
    axes[1, 2] = plt.subplot(2, 3, 6)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_array, axes[:, 0])
    ax_img.set_title("Low contrast image")
    ax_hist.set_ylabel("Number of pixels")

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
    ax_img.set_title("Global equalize")

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
    ax_img.set_title("Local equalize")
    ax_cdf.set_ylabel("Fraction of total intensity")

    # prevent overlap of y-axis labels
    fig.tight_layout()
    equalized_img = perform_adaptive_hist_equalization(img_array, 64, 48)

    plt.figure(figsize=(10, 10))
    plt.imshow(img_eq, cmap="gray")
    plt.figure(figsize=(10, 10))
    plt.imshow(
        perform_adaptive_hist_equalization(img_array, 64, 48),
        cmap="gray",
    )
    plt.show()
