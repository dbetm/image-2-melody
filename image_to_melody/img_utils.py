import random
from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans


def get_pixel_average_values(
    img: np.ndarray, sample_percent: float = 0.15, seed: int = 42
) -> Tuple[int, int, int]:
    """Sample the image using a uniform probability distribution and then
    compute the average for each channel, the image must have tree channels (example
    RGB and HSV). The order is keept."""
    height, width, _ = img.shape
    random.seed(seed)

    total_pixels = int((width * height) * sample_percent)
    first_ch_acc, second_ch_acc, third_ch_acc = 0, 0, 0

    for _ in range(total_pixels):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        pix = img[y][x]

        first_ch_acc += pix[0]
        second_ch_acc += pix[1]
        third_ch_acc += pix[2]

    avg_first = first_ch_acc // total_pixels
    avg_second = second_ch_acc // total_pixels
    avg_third = third_ch_acc // total_pixels

    return (avg_first, avg_second, avg_third)


def get_pixel_median_values(
    img: np.ndarray, sample_percent: float = 0.15, seed: int = 42
) -> Tuple[int, int, int]:
    """Randomly sample the pixels in the image using a uniform probability distribution. 
    Compute the median value for each channel in the sampled pixels. The image must 
    have three channels, such as RGB or HSV, and the order of channels must be maintained.
    """
    height, width, _ = img.shape
    random.seed(seed)

    total_pixels = int((width * height) * sample_percent)
    first_channel, second_channel, third_channel = list(), list(), list()

    for _ in range(total_pixels):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        pix = img[y][x]

        first_channel.append(pix[0])
        second_channel.append(pix[1])
        third_channel.append(pix[2])

    median_first = np.median(np.array(first_channel))
    median_second = np.median(np.array(second_channel))
    median_third = np.median(np.array(third_channel))

    return (median_first, median_second, median_third)


def get_k_representatives(pixels: np.array, k: int = 5) -> list:
    """Given an array (N, 3), compute the k cluster and then return the centroids
    which are representatives of the pixels group given."""
    k_means_cluster = KMeans(n_clusters=k, n_init=5).fit(pixels)
    centroids = k_means_cluster.cluster_centers_

    # Sort the centroids by their proximity to the origin
    centroids_distances = np.linalg.norm(centroids, axis=1)
    sorted_centroids_indices = np.argsort(centroids_distances)
    sorted_centroids = centroids[sorted_centroids_indices]

    return sorted_centroids


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