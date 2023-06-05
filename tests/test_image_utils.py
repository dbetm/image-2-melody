from math import isclose

import cv2
import numpy as np
import pytest

from image_to_melody.img_utils import (
    get_pixel_median_values,
    get_pixel_average_values,
    get_representative_pixels,
    resize_image,
)


@pytest.fixture
def black_square_image_300_x_300():
    width, height = 300, 300
    yield np.zeros((height, width, 3), dtype=np.uint8)


@pytest.fixture
def green_with_white_bar_image_200_x_200():
    """RGB values for green: 
    R=0.0
    G=171.3
    B=62.3
    """
    imag_path = "tests/data/images/green_background_with_white_bar.png"
    image = cv2.imread(filename=imag_path)

    yield cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)


@pytest.fixture
def board_di_color_image_200_x_200():
    """Image with four squares, two green and two red (interspersed).
    RGB values for green: 
    R=0.0
    G=171.3
    B=62.3

    RGB values for red:
    R=244.7
    G=8.0
    B=65.4
    """
    imag_path = "tests/data/images/board_di_color.png"
    image = cv2.imread(filename=imag_path)

    yield cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)


def test_get_pixel_average_values(black_square_image_300_x_300):
    avg_rgb_values = get_pixel_average_values(
        black_square_image_300_x_300, sample_percent=0.06
    )

    # check all channel values are near to 0
    for channel_id in range(3):
        assert avg_rgb_values[channel_id] < 1


def test_get_pixel_median_values(green_with_white_bar_image_200_x_200):
    median_rgb_values = get_pixel_median_values(green_with_white_bar_image_200_x_200)

    assert int(median_rgb_values[0]) == 0
    assert int(median_rgb_values[1]) == 171
    assert int(median_rgb_values[2]) == 62


def test_get_representative_pixels(board_di_color_image_200_x_200):
    number_slices = 2
    k = 2

    repr_pixels = get_representative_pixels(
        img=board_di_color_image_200_x_200,
        number_slices=number_slices,
        k_repr_by_slice=k,
    )

    assert len(repr_pixels) == (number_slices * k)

    # compare values of representative pixels of the areas with the same color
    for (val_1, val_2) in zip(repr_pixels[0], repr_pixels[2]):
        assert isclose(val_1, val_2, rel_tol=1.0)

    for (val_1, val_2) in zip(repr_pixels[1], repr_pixels[3]):
        assert isclose(val_1, val_2, rel_tol=1.0)


def test_resize_image_downscale(black_square_image_300_x_300):
    threshold_dim = 200

    resized_image = resize_image(
        black_square_image_300_x_300, threshold_dim=threshold_dim
    )

    assert threshold_dim == max(resized_image.shape)


def test_resize_image_keep_size(black_square_image_300_x_300):
    threshold_dim = 301

    resized_image = resize_image(
        black_square_image_300_x_300, threshold_dim=threshold_dim
    )

    assert black_square_image_300_x_300.shape == resized_image.shape
