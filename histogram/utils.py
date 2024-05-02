import numpy as np


class Quadrant:
    """
    Represents a quadrant in an image region for interpolation.
    """

    def __init__(self, upper_left, upper_right, bottom_left, bottom_right):
        """
        Initialize the quadrant with its four corner points.

        Parameters:
        - upper_left: Upper left corner of the quadrant.
        - upper_right: Upper right corner of the quadrant.
        - bottom_left: Bottom left corner of the quadrant.
        - bottom_right: Bottom right corner of the quadrant.
        """
        self.upper_left = upper_left
        self.upper_right = upper_right
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right

    def interpolate_pixel(self, i, j, transforms, img_array):
        """
        Interpolate the pixel value using bilinear interpolation.

        Parameters:
        - i, j: Pixel coordinates.
        - transforms: Dictionary of transformation functions for each region.
        - img_array: Input image array.

        Returns:
        - Interpolated pixel value.
        """
        R1 = transforms[self.upper_left][img_array[i, j]] * (
            (self.upper_right[1] - j) / (self.upper_right[1] - self.upper_left[1])
        ) + transforms[self.upper_right][img_array[i, j]] * (
            (j - self.upper_left[1]) / (self.upper_right[1] - self.upper_left[1])
        )
        R2 = transforms[self.bottom_left][img_array[i, j]] * (
            (self.bottom_right[1] - j) / (self.bottom_right[1] - self.bottom_left[1])
        ) + transforms[self.bottom_right][img_array[i, j]] * (
            (j - self.bottom_left[1]) / (self.bottom_right[1] - self.bottom_left[1])
        )

        return R1 * (
            (self.bottom_left[0] - i) / (self.bottom_left[0] - self.upper_left[0])
        ) + R2 * ((i - self.upper_left[0]) / (self.bottom_left[0] - self.upper_left[0]))


class Region:
    """
    Represents a region in an image with four quadrants.
    """

    def __init__(self, centre, region_len_h, region_len_w):
        """
        Initialize the region with its centre and dimensions.

        Parameters:
        - centre: Centre coordinates of the region.
        - region_len_h: Height of the region.
        - region_len_w: Width of the region.
        """
        self.centre = centre
        self.region_len_h = region_len_h
        self.region_len_w = region_len_w

        self.UL = Quadrant(
            upper_left=(centre[0] - region_len_h, centre[1] - region_len_w),
            upper_right=(centre[0] - region_len_h, centre[1]),
            bottom_left=(centre[0], centre[1] - region_len_w),
            bottom_right=(centre[0], centre[1]),
        )

        self.UR = Quadrant(
            upper_left=(centre[0] - region_len_h, centre[1]),
            upper_right=(centre[0] - region_len_h, centre[1] + region_len_w),
            bottom_left=(centre[0], centre[1]),
            bottom_right=(centre[0], centre[1] + region_len_w),
        )

        self.BL = Quadrant(
            upper_left=(centre[0], centre[1] - region_len_w),
            upper_right=(centre[0], centre[1]),
            bottom_left=(centre[0] + region_len_h, centre[1] - region_len_w),
            bottom_right=(centre[0] + region_len_h, centre[1]),
        )

        self.BR = Quadrant(
            upper_left=(centre[0], centre[1]),
            upper_right=(centre[0], centre[1] + region_len_w),
            bottom_left=(centre[0] + region_len_h, centre[1]),
            bottom_right=(centre[0] + region_len_h, centre[1] + region_len_w),
        )

    def process_quadrant(
        self,
        quadrant,
        transforms,
        img_array,
        equalized_img_array,
        with_neighbours=False,
    ):
        """
        Process a quadrant of the region and apply interpolation.

        Parameters:
        - quadrant: Quadrant identifier ("UL", "UR", "BL", "BR").
        - transforms: Dictionary of transformation functions for each region.
        - img_array: Input image array.
        - equalized_img_array: Array to store the equalized image.
        - with_neighbours: Flag to indicate if neighbouring quadrants should be considered.
        """
        if quadrant == "UL":
            for i in np.arange(self.centre[0] - self.region_len_h // 2, self.centre[0]):
                for j in np.arange(
                    self.centre[1] - self.region_len_w // 2, self.centre[1]
                ):
                    equalized_img_array[i, j] = (
                        self.UL.interpolate_pixel(i, j, transforms, img_array)
                        if with_neighbours
                        else transforms[self.centre][img_array[i, j]]
                    )

        if quadrant == "UR":
            for i in np.arange(self.centre[0] - self.region_len_h // 2, self.centre[0]):
                for j in np.arange(
                    self.centre[1], self.centre[1] + self.region_len_w // 2
                ):
                    equalized_img_array[i, j] = (
                        self.UR.interpolate_pixel(i, j, transforms, img_array)
                        if with_neighbours
                        else transforms[self.centre][img_array[i, j]]
                    )

        if quadrant == "BL":
            for i in np.arange(self.centre[0], self.centre[0] + self.region_len_h // 2):
                for j in np.arange(
                    self.centre[1] - self.region_len_w // 2, self.centre[1]
                ):
                    equalized_img_array[i, j] = (
                        self.BL.interpolate_pixel(i, j, transforms, img_array)
                        if with_neighbours
                        else transforms[self.centre][img_array[i, j]]
                    )

        if quadrant == "BR":
            for i in np.arange(self.centre[0], self.centre[0] + self.region_len_h // 2):
                for j in np.arange(
                    self.centre[1], self.centre[1] + self.region_len_w // 2
                ):
                    equalized_img_array[i, j] = (
                        self.BR.interpolate_pixel(i, j, transforms, img_array)
                        if with_neighbours
                        else transforms[self.centre][img_array[i, j]]
                    )


def get_histogram(img_array: np.ndarray) -> np.ndarray:
    """
    Calculate the histogram of an input image.

    Parameters:
    - img_array: Input image array.

    Returns:
    - Histogram array.
    """
    levels = np.arange(256)
    occurences = np.zeros(256)

    for level in levels:
        occurences[level] = np.sum(img_array == level)
    return occurences / np.sum(occurences)
