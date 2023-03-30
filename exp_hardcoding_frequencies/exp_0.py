from typing import Callable, Union

import cv2
import numpy as np
import pandas as pd

from image_to_melody.musical_scale_freqs import *
from image_to_melody.img_utils import (
    get_representative_pixels,
    get_rgb_average_values,
)


EXPERIMENT_ID = 0
# IMAGE PROCESSING
NUMBER_SLICES = 25
K = 5 # number of representative colors by slice
# MELODY GENERATION
SAMPLE_RATE = 44100 # 44.1 KHz - standard used in most CDs and digital audio formats
T = 0.2 # second duration


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


def build_playable_audio(df: pd.DataFrame, gen_singal_fn: Callable) -> np.ndarray:
    frequencies = df["notes"].to_numpy()
    octaves = df["octave"].to_numpy()

    # t represents an array of int(T*sample_rate) time values starting from 0 
    # and ending at T, with a fixed duration between each sample
    t = np.linspace(0, T, int(T*SAMPLE_RATE), endpoint=False)

    song = np.array([])

    for octave, freq, in zip(octaves, frequencies):
        val = freq * octave
        note  = 1 * gen_singal_fn(2*np.pi*val*t) # Represent each note as a sign wave
        song  = np.concatenate([song, note]) # Add notes into song array to make song

    return song


def image_to_melody(
    image: np.ndarray, gen_wave_fn: Callable = np.sin
):
    rgb_img = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2HSV)

    # height, width, channels = hsv_img.shape
    all_representative_pixels = get_representative_pixels(hsv_img, NUMBER_SLICES, K)

    df_repixels = pd.DataFrame(
        all_representative_pixels, columns=["hue", "saturation", "value"]
    )

    print(f"Number of representative pixels: {df_repixels.shape[0]}")

    avg_r, avg_g, avg_b = get_rgb_average_values(rgb_img)

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

    audio = build_playable_audio(df_repixels, gen_wave_fn)

    return audio


if __name__ == "__main__":
    pass