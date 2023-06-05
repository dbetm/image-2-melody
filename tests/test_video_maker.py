import os

import cv2
import pytest

from image_to_melody.video_maker import (
    DELTA_BRIGHT, highlight_pixel, create_video
)


@pytest.mark.skip(reason="requires more complex testing [implement later]")
def test_create_video():
    video_path = os.path.join("outputs", "exp_3", "final.mp4")
    img = cv2.imread(filename="sample_images/003_starry_night.jpg")
    audio_path = "experiments/tmp/melody-20230524-21:44:08.mp3"

    video_filepath = create_video(
        img=img,
        n_slices=5,
        audio_path=audio_path,
        output_path=video_path,
        rate_img_repetition=5,
        fps=5,
    )

    # delete tmp video without audio
    os.remove(video_path)


@pytest.mark.parametrize(
    "intensity, is_boder, expected",
    [
        (255, False, 255),
        (100, False, 100 + DELTA_BRIGHT),
        (2, True, 3)
    ]
)
def test_highlight_pixel(intensity, is_boder, expected):
    assert highlight_pixel(intensity, is_border=is_boder) == expected

