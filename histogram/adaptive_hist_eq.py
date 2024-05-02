import numpy as np
from typing import Dict, Tuple
from histogram.global_hist_eq import get_equalization_transform_of_img
from histogram.utils import Region


def calculate_eq_transformations_of_regions(
    img_array: np.ndarray, region_len_h: int, region_len_w: int
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Calculate equalization transformations for each region in the image.

    Parameters:
    - img_array: Input image array (2D numpy array).
    - region_len_h: Height of the region.
    - region_len_w: Width of the region.

    Returns:
    - region_to_eq_transform: Dictionary with region centres as keys and their
      corresponding equalization transformations as values.
    """
    H, W = img_array.shape
    num_regions_H = H // region_len_h
    num_regions_W = W // region_len_w

    region_to_eq_transform = {}

    # Generate coordinates for the centers of each region
    centres = [
        (x, y)
        for x in np.linspace(
            0 + region_len_h // 2, H - region_len_h // 2, num_regions_H, dtype=int
        )
        for y in np.linspace(
            0 + region_len_w // 2, W - region_len_w // 2, num_regions_W, dtype=int
        )
    ]

    # Calculate equalization transformation for each region
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
    """
    Perform adaptive histogram equalization on the input image.

    Parameters:
    - img_array: Input image array (2D numpy array).
    - region_len_h: Height of the region.
    - region_len_w: Width of the region.

    Returns:
    - equalized_img: Equalized image array (2D numpy array).
    """
    copy_img_array = img_array.copy()

    H, W = copy_img_array.shape

    num_regions_H = H // region_len_h
    num_regions_W = W // region_len_w

    new_H = num_regions_H * region_len_h
    new_W = num_regions_W * region_len_w

    # Crop the image to have dimensions as multiples of region lengths
    cropped_img_array = copy_img_array[
        (H - new_H) // 2 : (H - new_H) // 2 + new_H,
        (W - new_W) // 2 : (W - new_W) // 2 + new_W,
    ]

    # Initialize an empty array for the equalized image
    equalized_img = np.zeros((new_H, new_W))

    # Calculate equalization transformations for each region
    region_to_eq_transform = calculate_eq_transformations_of_regions(
        cropped_img_array, region_len_h, region_len_w
    )

    # Process each region and its quadrants
    for index, ((centre_h, centre_w), T) in enumerate(region_to_eq_transform.items()):
        current_region = Region((centre_h, centre_w), region_len_h, region_len_w)

        # Determine which quadrant to process based on the region's center
        # and apply histogram equalization
        # Note: The conditions for each quadrant are based on the relative position
        # of the region's center to the image's boundaries.
        # 'process_quadrant' is assumed to be a method in the 'Region' class.
        # The 'with_neighbors' parameter determines whether to consider neighboring regions.
        # The quadrants are named as: UL (Upper Left), UR (Upper Right), BL (Bottom Left),
        # and BR (Bottom Right).

        # Process top-left region
        if current_region.centre == (region_len_h // 2, region_len_w // 2):
            for quadrant, with_neighbors in [
                ("UL", False),
                ("UR", False),
                ("BL", False),
                ("BR", True),
            ]:
                current_region.process_quadrant(
                    quadrant=quadrant,
                    transforms=region_to_eq_transform,
                    img_array=cropped_img_array,
                    equalized_img_array=equalized_img,
                    with_neighbours=with_neighbors,
                )

        # Process top-right region
        if current_region.centre == (region_len_h // 2, new_W - region_len_w // 2):
            for quadrant, with_neighbors in [
                ("UL", False),
                ("UR", False),
                ("BL", True),
                ("BR", False),
            ]:
                current_region.process_quadrant(
                    quadrant=quadrant,
                    transforms=region_to_eq_transform,
                    img_array=cropped_img_array,
                    equalized_img_array=equalized_img,
                    with_neighbours=with_neighbors,
                )

        # Process top region except top-left and top-right
        if index > 0 and index < num_regions_W - 1:
            for quadrant, with_neighbors in [
                ("UL", False),
                ("UR", False),
                ("BL", True),
                ("BR", True),
            ]:
                current_region.process_quadrant(
                    quadrant=quadrant,
                    transforms=region_to_eq_transform,
                    img_array=cropped_img_array,
                    equalized_img_array=equalized_img,
                    with_neighbours=with_neighbors,
                )

        # Process bottom-left region
        if current_region.centre == (new_H - region_len_h // 2, region_len_w // 2):
            for quadrant, with_neighbors in [
                ("UL", False),
                ("UR", True),
                ("BL", False),
                ("BR", False),
            ]:
                current_region.process_quadrant(
                    quadrant=quadrant,
                    transforms=region_to_eq_transform,
                    img_array=cropped_img_array,
                    equalized_img_array=equalized_img,
                    with_neighbours=with_neighbors,
                )

        # Process bottom-right region
        if current_region.centre == (
            new_H - region_len_h // 2,
            new_W - region_len_w // 2,
        ):
            for quadrant, with_neighbors in [
                ("UL", True),
                ("UR", False),
                ("BL", False),
                ("BR", False),
            ]:
                current_region.process_quadrant(
                    quadrant=quadrant,
                    transforms=region_to_eq_transform,
                    img_array=cropped_img_array,
                    equalized_img_array=equalized_img,
                    with_neighbours=with_neighbors,
                )

        # Process regions at the image boundaries
        # Here, different conditions are checked to determine which quadrants to process
        # based on the region's index and position in the image.
        # Each condition corresponds to a specific region in the image boundary.

        # Process regions at the middle of the image (not at the boundaries)
        if (
            index > (num_regions_H - 1) * num_regions_W
            and index < num_regions_W * num_regions_H - 1
        ):
            for quadrant, with_neighbors in [
                ("UL", True),
                ("UR", True),
                ("BL", False),
                ("BR", False),
            ]:
                current_region.process_quadrant(
                    quadrant=quadrant,
                    transforms=region_to_eq_transform,
                    img_array=cropped_img_array,
                    equalized_img_array=equalized_img,
                    with_neighbours=with_neighbors,
                )

        # Process leftmost regions
        if (
            index % num_regions_W == 0
            and index != 0
            and index != (num_regions_H - 1) * num_regions_W
        ):
            for quadrant, with_neighbors in [
                ("UL", False),
                ("UR", True),
                ("BL", False),
                ("BR", True),
            ]:
                current_region.process_quadrant(
                    quadrant=quadrant,
                    transforms=region_to_eq_transform,
                    img_array=cropped_img_array,
                    equalized_img_array=equalized_img,
                    with_neighbours=with_neighbors,
                )

        # Process rightmost regions
        if (
            index % num_regions_W == num_regions_W - 1
            and index > num_regions_W - 1
            and index < num_regions_H * num_regions_W - 1
        ):
            for quadrant, with_neighbors in [
                ("UL", True),
                ("UR", False),
                ("BL", True),
                ("BR", False),
            ]:
                current_region.process_quadrant(
                    quadrant=quadrant,
                    transforms=region_to_eq_transform,
                    img_array=cropped_img_array,
                    equalized_img_array=equalized_img,
                    with_neighbours=with_neighbors,
                )

        # Process regions that are not at the boundaries
        if (
            current_region.centre[0] > current_region.region_len_h // 2
            and current_region.centre[0] < new_H - current_region.region_len_h // 2
        ) and (
            current_region.centre[1] > current_region.region_len_w // 2
            and current_region.centre[1] < new_W - current_region.region_len_w // 2
        ):
            for quadrant, with_neighbors in [
                ("UL", True),
                ("UR", True),
                ("BL", True),
                ("BR", True),
            ]:
                current_region.process_quadrant(
                    quadrant=quadrant,
                    transforms=region_to_eq_transform,
                    img_array=cropped_img_array,
                    equalized_img_array=equalized_img,
                    with_neighbours=with_neighbors,
                )

    return equalized_img
