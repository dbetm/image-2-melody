import os
import random
from typing import Callable, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import square
from tqdm import tqdm

from musical_scale_freqs import *
from splitter import get_k_representatives


BASE_OUTPUT_PATH = "../output_music/exp_1/"
IMAGES_PATH = "../sample_images/"
EXPERIMENT_ID = 0
# IMAGE PROCESSING
NUMBER_SLICES = 25
K = 5 # number of representative colors by slice
# MELODY GENERATION
SAMPLE_RATE = 44100 # 44.1 KHz - standard used in most CDs and digital audio formats
T = 0.2 # second duration


def get_representative_pixels(hsv_img: np.ndarray) -> list:
    """Iterate over all the slices, computing for each one the K 
    representative colors K / (height * slice_size) percent.
    """
    height, width, _ = hsv_img.shape

    assert width > NUMBER_SLICES
    slice_width = width // NUMBER_SLICES

    all_representative_pixels = list()

    for slice_i in range(0, width, slice_width):
        # iterate a single slice to get the HSV values
        hsv_values = list()
        for x in range(slice_i, min(slice_i + slice_width, width)):
            for y in range(height):
                hsv_values.append(hsv_img[y][x])

        # compute the K representative pixels
        representatives = get_k_representatives(pixels=hsv_values, k=K)
        representatives = representatives.tolist()

        for representative in representatives:
            all_representative_pixels.append(representative)

    return all_representative_pixels


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


def image_to_melody(image: np.ndarray, output_path: str, gen_wave_fn: Callable) -> None:
    rgb_img = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2HSV)

    # height, width, channels = hsv_img.shape
    all_representative_pixels = get_representative_pixels(hsv_img)

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

    # save audio 
    wavfile.write( 
        output_path,
        rate=SAMPLE_RATE,
        data=audio.astype(np.float32)
    )


if __name__ == "__main__":
    images_filenames = os.listdir(path=IMAGES_PATH)
    images_filenames.sort()
    # images_filenames = ["009_turkey_nebula.jpg"]

    sin_fn = np.sin
    square_fn = square


    for filename in tqdm(images_filenames):
        if filename.startswith("."):
            continue

        full_path = os.path.join(IMAGES_PATH, filename)
    
        if not os.path.isfile(full_path):
            continue

        print(filename)

        img = cv2.imread(filename=full_path)
        audio_filename = filename.split(".")[0]
        audio_filename += ".wav"

        image_to_melody(
            img,
            output_path=os.path.join(BASE_OUTPUT_PATH, audio_filename),
            gen_wave_fn=square_fn,
        )
