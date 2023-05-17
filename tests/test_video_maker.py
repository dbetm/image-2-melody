import pytest

from image_to_melody.video_maker import highlight_pixel


def test_highlight_pixel():
    assert highlight_pixel(255, is_border=False) == 255

