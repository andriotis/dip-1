import numpy as np
import global_hist_eq as ghe


def calculate_eq_transformations_of_regions(
    img_array: np.ndarray, region_len_h: int, region_len_w: int
) -> dict[tuple[int, int], np.ndarray]:

    H, W = img_array.shape

    region_to_eq_transform = {}

    origins = [
        (i, j)
        for i in np.arange(0, H, region_len_h)
        for j in np.arange(0, W, region_len_w)
    ]

    for origin_h, origin_w in origins:

        region_to_eq_transform[(origin_h, origin_w)] = (
            ghe.get_equalization_transform_of_img(
                img_array[
                    origin_h : origin_h + region_len_h,
                    origin_w : origin_w + region_len_w,
                ]
            )
        )

    return region_to_eq_transform
