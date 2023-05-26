import os

import cv2
import pytest

from image_to_melody.video_maker import highlight_pixel, create_video


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


def test_highlight_pixel():
    assert highlight_pixel(255, is_border=False) == 255

