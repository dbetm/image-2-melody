import random
from typing import Tuple

import numpy as np

from image_to_melody.cluster import get_k_representatives


def get_rgb_average_values(
    rgb_img: np.ndarray, sample_percent: float = 0.15
) -> Tuple[int, int, int]:
    """Sample the image using a uniform probability distribution and then
    compute the average for each channel (RGB)."""
    height, width, _ = rgb_img.shape

    total_pixels = int((width * height) * sample_percent)
    r_acc, g_acc, b_acc = 0, 0, 0

    for _ in range(total_pixels):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        pix = rgb_img[y][x]

        r_acc += pix[0]
        g_acc += pix[1]
        b_acc += pix[2]

    avg_r = r_acc // total_pixels
    avg_g = g_acc // total_pixels
    avg_b = b_acc // total_pixels

    return (avg_r, avg_g, avg_b)


def get_representative_pixels(
    img: np.ndarray, number_slices: int, k_repr_by_slice: int
) -> list:
    """Iterate over all the slices, computing for each one the K 
    representative colors K / (height * slice_size) percent.
    """
    height, width, _ = img.shape

    assert width > number_slices
    slice_width = width // number_slices

    all_representative_pixels = list()

    for slice_i in range(0, width, slice_width):
        # iterate a single slice to get the HSV values
        channel_values = list()
        for x in range(slice_i, min(slice_i + slice_width, width)):
            for y in range(height):
                channel_values.append(img[y][x])

        # compute the K representative pixels
        representatives = get_k_representatives(
            pixels=channel_values, k=k_repr_by_slice
        )
        representatives = representatives.tolist()

        for representative in representatives:
            all_representative_pixels.append(representative)

    return all_representative_pixels