import numpy as np
from typing import Tuple
from PIL import Image

class HistogramEqualization:
    def __init__(self, img_array: np.ndarray):
        self.img_array = img_array
        self.H, self.W = img_array.shape

    def get_equalization_transform(self) -> np.ndarray:
        L = 256
        levels = np.arange(L)
        occurences = np.zeros(L)

        for level in levels:
            occurences[level] = np.sum(self.img_array == level)
        pixel_probability = occurences / np.sum(occurences)

        v = np.array([np.sum(pixel_probability[: level + 1]) for level in levels])

        y = np.round(((v - v[0]) / (1 - v[0])) * (L - 1))

        equalization_transform = y.astype(np.uint8)

        return equalization_transform

    def perform_global_hist_equalization(self) -> np.ndarray:
        equalization_transform = self.get_equalization_transform()
        equalized_img = equalization_transform[self.img_array]
        return equalized_img

    def calculate_eq_transformations_of_regions(self, region_len_h: int, region_len_w: int) -> dict:
        num_regions_H = self.H // region_len_h
        num_regions_W = self.W // region_len_w

        region_to_eq_transform = {}

        centres = [
            (x, y)
            for x in np.linspace(
                0 + region_len_h // 2, self.H - region_len_h // 2, num_regions_H, dtype=int
            )
            for y in np.linspace(
                0 + region_len_w // 2, self.W - region_len_w // 2, num_regions_W, dtype=int
            )
        ]

        for centre_h, centre_w in centres:
            region_to_eq_transform[(centre_h, centre_w)] = self.get_equalization_transform_of_region(
                centre_h, centre_w, region_len_h, region_len_w
            )

        return region_to_eq_transform

    def get_equalization_transform_of_region(self, centre_h, centre_w, region_len_h, region_len_w):
        return self.get_equalization_transform(
            self.img_array[centre_h - region_len_h // 2 : centre_h + region_len_h // 2,centre_w - region_len_w // 2 : centre_w + region_len_w // 2]
        )

    def perform_adaptive_hist_equalization(self, region_len_h: int, region_len_w: int) -> np.ndarray:
        copy_img_array = self.img_array.copy()

        new_H = (self.H // region_len_h) * region_len_h
        new_W = (self.W // region_len_w) * region_len_w

        cropped_img_array = copy_img_array[
            (self.H - new_H) // 2 : (self.H - new_H) // 2 + new_H,
            (self.W - new_W) // 2 : (self.W - new_W) // 2 + new_W,
        ]

        equalized_img = np.zeros((new_H, new_W))

        region_to_eq_transform = self.calculate_eq_transformations_of_regions(region_len_h, region_len_w)

        for index, ((centre_h, centre_w), T) in enumerate(region_to_eq_transform.items()):
            equalized_img = self.process_region(
                equalized_img,
                cropped_img_array,
                region_to_eq_transform,
                centre_h,
                centre_w,
                region_len_h,
                region_len_w,
                index,
                new_H // region_len_h,
                new_W // region_len_w,
            )

        return equalized_img

    def process_top_left_region(self, equalized_img, cropped_img_array, region_to_eq_transform, centre_h, centre_w, region_len_h, region_len_w):
            T = region_to_eq_transform[(centre_h, centre_w)]
            for j in np.arange(centre_w - region_len_w // 2, centre_w + region_len_w // 2):
        for i in np.arange(centre_h - region_len_h // 2, centre_h):
            equalized_img[i, j] = T[cropped_img_array[i, j]]
        for i in np.arange(centre_h, centre_h + region_len_h // 2):
            equalized_img[i, j] = T[cropped_img_array[i, j]]
        for i in np.arange(centre_h, centre_h + region_len_h // 2):
            for j in np.arange(centre_w, centre_w + region_len_w // 2):
                equalized_img = self.interpolate_quadrants(
                    equalized_img,
                    cropped_img_array,
                    region_to_eq_transform,
                    centre_h,
                    centre_w,
                    region_len_h,
                    region_len_w,
                    i,
                    j,
                )

    return equalized_img

    def process_top_right_region(self, equalized_img, cropped_img_array, region_to_eq_transform, centre_h, centre_w, region_len_h, region_len_w):
        T = region_to_eq_transform[(centre_h, centre_w)]
        for j in np.arange(centre_w - region_len_w // 2, centre_w + region_len_w // 2):
            for i in np.arange(centre_h - region_len_h // 2, centre_h):
                equalized_img[i, j] = T[cropped_img_array[i, j]]
        for j in np.arange(centre_w, centre_w + region_len_w // 2):
            for i in np.arange(centre_h, centre_h + region_len_h // 2):
                equalized_img[i, j] = T[cropped_img_array[i, j]]

        for i in np.arange(centre_h - region_len_h // 2, centre_h):
            for j in np.arange(centre_w - region_len_w // 2, centre_w):
                equalized_img = self.interpolate_quadrants(
                    equalized_img,
                    cropped_img_array,
                    region_to_eq_transform,
                    centre_h,
                    centre_w,
                    region_len_h,
                    region_len_w,
                    i,
                    j,
                )

    return equalized_img

    def process_bottom_left_region(self, equalized_img, cropped_img_array, region_to_eq_transform, centre_h, centre_w, region_len_h, region_len_w):
        T = region_to_eq_transform[(centre_h, centre_w)]
        for i in np.arange(centre_h - region_len_h // 2, centre_h):
            for j in np.arange(centre_w - region_len_w // 2, centre_w):
                equalized_img[i, j] = T[cropped_img_array[i, j]]
            for j in np.arange(centre_w, centre_w + region_len_w // 2):
                equalized_img = self.interpolate_quadrants(
                    equalized_img,
                    cropped_img_array,
                    region_to_eq_transform,
                    centre_h,
                    centre_w,
                    region_len_h,
                    region_len_w,
                    i,
                    j,
                )

        for i in np.arange(centre_h, centre_h + region_len_h // 2):
            for j in np.arange(centre_w - region_len_w // 2, centre_w + region_len_w // 2):
                equalized_img[i, j] = T[cropped_img_array[i, j]]

    return equalized_img

    def process_bottom_right_region(self, equalized_img, cropped_img_array, region_to_eq_transform, centre_h, centre_w, region_len_h, region_len_w):
        T = region_to_eq_transform[(centre_h, centre_w)]
        for i in np.arange(centre_h - region_len_h // 2, centre_h):
            for j in np.arange(centre_w - region_len_w // 2, centre_w):
                equalized_img = self.interpolate_quadrants(
                    equalized_img,
                    cropped_img_array,
                    region_to_eq_transform,
                    centre_h,
                    centre_w,
                    region_len_h,
                    region_len_w,
                    i,
                    j,
                )
            for j in np.arange(centre_w, centre_w + region_len_w // 2):
                equalized_img[i, j] = T[cropped_img_array[i, j]]

        for i in np.arange(centre_h, centre_h + region_len_h // 2):
            for j in np.arange(centre_w - region_len_w // 2, centre_w + region_len_w // 2):
                equalized_img[i, j] = T[cropped_img_array[i, j]]

    return equalized_img

    def process_middle_region(self, equalized_img, cropped_img_array, region_to_eq_transform, centre_h, centre_w, region_len_h, region_len_w):
        T = region_to_eq_transform[(centre_h, centre_w)]
        for i in np.arange(centre_h - region_len_h // 2, centre_h):
            for j in np.arange(centre_w - region_len_w // 2, centre_w + region_len_w // 2):
                equalized_img[i, j] = T[cropped_img_array[i, j]]

        for i in np.arange(centre_h, centre_h + region_len_h // 2):
            for j in np.arange(centre_w - region_len_w // 2, centre_w):
                equalized_img = self.interpolate_quadrants(
                    equalized_img,
                    cropped_img_array,
                    region_to_eq_transform,
                    centre_h,
                    centre_w,
                    region_len_h,
                    region_len_w,
                    i,
                    j,
                )
            for j in np.arange(centre_w, centre_w + region_len_w // 2):
                equalized_img = self.interpolate_quadrants(
                    equalized_img,
                    cropped_img_array,
                    region_to_eq_transform,
                    centre_h,
                    centre_w,
                    region_len_h,
                    region_len_w,
                    i,
                    j,
                )

    return equalized_img

    def process_bottom_middle_region(self, equalized_img, cropped_img_array, region_to_eq_transform, centre_h, centre_w, region_len_h, region_len_w):
        T = region_to_eq_transform[(centre_h, centre_w)]
        for i in np.arange(centre_h, centre_h + region_len_h // 2):
            for j in np.arange(centre_w - region_len_w // 2, centre_w + region_len_w // 2):
                equalized_img[i, j] = T[cropped_img_array[i, j]]

        for i in np.arange(centre_h - region_len_h // 2, centre_h):
            for j in np.arange(centre_w - region_len_w // 2, centre_w):
                equalized_img = self.interpolate_quadrants(
                    equalized_img,
                    cropped_img_array,
                    region_to_eq_transform,
                    centre_h,
                    centre_w,
                    region_len_h,
                    
                    
                    region_len_w,
                i,
                j,
            )
        for j in np.arange(centre_w, centre_w + region_len_w // 2):
            equalized_img = self.interpolate_quadrants(
                equalized_img,
                cropped_img_array,
                region_to_eq_transform,
                centre_h,
                centre_w,
                region_len_h,
                region_len_w,
                i,
                j,
            )

return equalized_img

def process_left_middle_region(self, equalized_img, cropped_img_array, region_to_eq_transform, centre_h, centre_w, region_len_h, region_len_w):
    T = region_to_eq_transform[(centre_h, centre_w)]
    for j in np.arange(centre_w - region_len_w // 2, centre_w):
        for i in np.arange(centre_h - region_len_h // 2, centre_h + region_len_h // 2):
            equalized_img[i, j] = T[cropped_img_array[i, j]]

    for i in np.arange(centre_h - region_len_h // 2, centre_h):
        for j in np.arange(centre_w, centre_w + region_len_w // 2):
            equalized_img = self.interpolate_quadrants(
                equalized_img,
                cropped_img_array,
                region_to_eq_transform,
                centre_h,
                centre_w,
                region_len_h,
                region_len_w,
                i,
                j,
            )

    for i in np.arange(centre_h, centre_h + region_len_h // 2):
        for j in np.arange(centre_w, centre_w + region_len_w // 2):
            equalized_img = self.interpolate_quadrants(
                equalized_img,
                cropped_img_array,
                region_to_eq_transform,
                centre_h,
                centre_w,
                region_len_h,
                region_len_w,
                i,
                j,
            )

return equalized_img

def process_right_middle_region(self, equalized_img, cropped_img_array, region_to_eq_transform, centre_h, centre_w, region_len_h, region_len_w):
    T = region_to_eq_transform[(centre_h, centre_w)]
    for j in np.arange(centre_w, centre_w + region_len_w // 2):
        for i in np.arange(centre_h - region_len_h // 2, centre_h + region_len_h // 2):
            equalized_img[i, j] = T[cropped_img_array[i, j]]

    for i in np.arange(centre_h - region_len_h // 2, centre_h):
        for j in np.arange(centre_w - region_len_w // 2, centre_w):
            equalized_img = self.interpolate_quadrants(
                equalized_img,
                cropped_img_array,
                region_to_eq_transform,
                centre_h,
                centre_w,
                region_len_h,
                region_len_w,
                i,
                j,
            )

    for i in np.arange(centre_h, centre_h + region_len_h // 2):
        for j in np.arange(centre_w - region_len_w // 2, centre_w):
            equalized_img = self.interpolate_quadrants(
                equalized_img,
                cropped_img_array,
                region_to_eq_transform,
                centre_h,
                centre_w,
                region_len_h,
                region_len_w,
                i,
                j,
            )

return equalized_img

def process_interior_region(self, equalized_img, cropped_img_array, region_to_eq_transform, centre_h, centre_w, region_len_h, region_len_w):
    for i in np.arange(centre_h - region_len_h // 2, centre_h):
        for j in np.arange(centre_w - region_len_w // 2, centre_w):
            equalized_img = self.interpolate_quadrants(
                equalized_img,
                cropped_img_array,
                region_to_eq_transform,
                centre_h,
                centre_w,
                region_len_h,
                region_len_w,
                i,
                j,
            )
        for j in np.arange(centre_w, centre_w + region_len_w // 2):
            equalized_img = self.interpolate_quadrants(
                equalized_img,
                cropped_img_array,
                region_to_eq_transform,
                centre_h,
                centre_w,
                region_len_h,
                region_len_w,
                i,
                j,
            )

    for i in np.arange(centre_h, centre_h + region_len_h // 2):
        for j in np.arange(centre_w - region_len_w // 2, centre_w):
            equalized_img = self.interpolate_quadrants(
                equalized_img,
                cropped_img_array,
                region_to_eq_transform,
                centre_h,
                centre_w,
                region_len_h,
                region_len_w,
                i,
                j,
            )
        for j in np.arange(centre_w, centre_w + region_len_w // 2):
            equalized_img = self.interpolate_quadrants(
                equalized_img,
                cropped_img_array,
                region_to_eq_transform,
                centre_h,
                centre_w,
                region_len_h,
                region_len_w,
                i,
                j,
            )

return equalized_img
    def process_region(
        self,
        equalized_img,
        cropped_img_array,
        region_to_eq_transform,
        centre_h,
        centre_w,
        region_len_h,
        region_len_w,
        index,
        num_regions_H,
        num_regions_W,
    ):
        if index == 0:
            equalized_img = self.process_top_left_region(
                equalized_img, cropped_img_array, region_to_eq_transform, centre_h, centre_w, region_len_h, region_len_w
            )
        elif index == num_regions_W - 1:
            equalized_img = self.process_top_right_region(
                equalized_img, cropped_img_array, region_to_eq_transform, centre_h, centre_w, region_len_h, region_len_w
            )
        elif index == (num_regions_H - 1) * num_regions_W:
            equalized_img = self.process_bottom_left_region(
                equalized_img, cropped_img_array, region_to_eq_transform, centre_h, centre_w, region_len_h, region_len_w
            )
        elif index == num_regions_W * num_regions_H - 1:
            equalized_img = self.process_bottom_right_region(
                equalized_img, cropped_img_array, region_to_eq_transform, centre_h, centre_w, region_len_h, region_len_w
            )
        elif index > 0 and index < num_regions_W - 1:
            equalized_img = self.process_middle_region(
                equalized_img, cropped_img_array, region_to_eq_transform, centre_h, centre_w, region_len_h, region_len_w
            )
        elif (index > (num_regions_H - 1) * num_regions_W and index < num_regions_W * num_regions_H - 1):
            equalized_img = self.process_bottom_middle_region(
                equalized_img, cropped_img_array, region_to_eq_transform, centre_h, centre_w, region_len_h, region_len_w
            )
        elif (index % num_regions_W == 0 and index != 0 and index != (num_regions_H - 1) * num_regions_W):
            equalized_img = self.process_left_middle_region(
                equalized_img, cropped_img_array, region_to_eq_transform, centre_h, centre_w, region_len_h, region_len_w
            )
        elif (
            index % num_regions_W == num_regions_W - 1
            and index > num_regions_W - 1
            and index < num_regions_H * num_regions_W - 1
        ):
            equalized_img = self.process_right_middle_region(
                equalized_img, cropped_img_array, region_to_eq_transform, centre_h, centre_w, region_len_h, region_len_w
            )
        else:
            equalized_img = self.process_interior_region(
                equalized_img, cropped_img_array, region_to_eq_transform, centre_h, centre_w, region_len_h, region_len_w
            )

        return equalized_img

        

def interpolate_quadrants(
    self,
    equalized_img,
    cropped_img_array,
    region_to_eq_transform,
    centre_h,
    centre_w,
    region_len_h,
    region_len_w,
    i,
    j,
):
    upper_left = (centre_h - region_len_h, centre_w - region_len_w)
    upper_right = (centre_h - region_len_h, centre_w + region_len_w)
    bottom_left = (centre_h + region_len_h, centre_w - region_len_w)
    bottom_right = (centre_h + region_len_h, centre_w + region_len_w)

    R1 = (
        region_to_eq_transform[upper_left][cropped_img_array[i, j]]
        * ((upper_right[1] - j) / (upper_right[1] - upper_left[1]))
        + region_to_eq_transform[upper_right][cropped_img_array[i, j]]
        * ((j - upper_left[1]) / (upper_right[1] - upper_left[1]))
    )
    R2 = (
        region_to_eq_transform[bottom_left][cropped_img_array[i, j]]
        * ((bottom_right[1] - j) / (bottom_right[1] - bottom_left[1]))
        + region_to_eq_transform[bottom_right][cropped_img_array[i, j]]
        * ((j - bottom_left[1]) / (bottom_right[1] - bottom_left[1]))
    )

    equalized_img[i, j] = (
        R1 * ((bottom_left[0] - i) / (bottom_left[0] - upper_left[0]))
        + R2 * ((i - upper_left[0]) / (bottom_left[0] - upper_left[0]))
    )

return equalized_img