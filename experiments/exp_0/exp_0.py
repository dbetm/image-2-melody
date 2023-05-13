"""In this experiment the frequency will be selected from groups of frequencies which
are known as musical scales. The HSV values are extracted from the image.
"""
from typing import Callable, Union

import cv2
import numpy as np
import pandas as pd
from pedalboard import Compressor, HighpassFilter

from image_to_melody.audio_processor import *
from image_to_melody.img_utils import (
    get_representative_pixels,
    get_pixel_average_values,
)


class Conf:
    # GENERAL
    EXPERIMENT_ID = 0
    # IMAGE PROCESSING
    NUMBER_SLICES = 25
    K = 5 # number of representative colors by slice
    # MELODY GENERATION
    SAMPLE_RATE = 44100 # 44.1 KHz - standard used in most CDs and digital audio formats
    T = 0.2 # second duration
    SOUND_EFFECTS = [
        HighpassFilter(),
        Compressor(threshold_db=0, ratio=25),
    ]
    # VIDEO GENERATION
    RATE_IMG_REPETITION = 2


def map_value_to_dest(
    val_: Union[int, float], max_threshold: int, dest_vals: list
) -> Union[int, float]:
    # generate equidistant thresholds
    thresholds = [
        threshold
        for threshold in 
        range(0, max_threshold, max_threshold // len(dest_vals))
    ]
    dest_value = dest_vals[-1]

    for i in range(1, len(thresholds)):
        if val_ < thresholds[i]:
            dest_value = dest_vals[i-1]
            break

    return dest_value


def image_to_melody(
    image: np.ndarray, gen_wave_fn: Callable = np.sin
):
    rgb_img = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2HSV)

    # height, width, channels = hsv_img.shape
    all_representative_pixels = get_representative_pixels(
        hsv_img, Conf.NUMBER_SLICES, Conf.K
    )

    df_repixels = pd.DataFrame(
        all_representative_pixels, columns=["hue", "saturation", "value"]
    )

    print(f"Number of representative pixels: {df_repixels.shape[0]}")

    avg_r, avg_g, avg_b = get_pixel_average_values(rgb_img)

    # select musical scale
    scale_freq = []

    if avg_r > 127 and avg_g > 127 and avg_b > 127:
        scale_freq = A_NATURAL_MINOR
        print("A natural minor")
    elif avg_r > avg_g and avg_r > avg_b:
        scale_freq = A_HARMONIC_MINOR
        print("A harmonic minor")
    elif avg_b > avg_r and avg_b > avg_g:
        scale_freq = A_MAJOR
        print("A major")
    else:
        scale_freq = A_HARMONIC_MAJOR
        print("A harmonic major")
    
    # Compute frequency of notes
    df_repixels["notes"] = df_repixels.apply(
        lambda row : map_value_to_dest(row["hue"], 180, scale_freq), axis=1
    )

    # Compute octaves
    octave_values = [0.5, 1, 2]
    df_repixels["octave"] = df_repixels.apply(
        lambda row : map_value_to_dest(row["saturation"], 255, octave_values), axis=1
    )

    # print stats
    print("-"*24)
    print(df_repixels.describe())
    print("-"*24)

    return build_playable_audio(
        df_repixels,
        gen_wave_fn,
        sample_rate=Conf.SAMPLE_RATE,
        time=Conf.T,
    )
