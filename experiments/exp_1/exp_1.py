"""In this experiment the frequency will be selected from a unique large group
of frequencies ordered in ascending order. Using the average RGB values.
"""
import cv2
import numpy as np
import pandas as pd
from pedalboard import Chorus, Delay, Reverb

from image_to_melody.audio_processor import build_playable_audio
from image_to_melody.img_utils import get_representative_pixels


class Conf:
    # GENERAL
    EXPERIMENT_ID = 1
    # IMAGE PROCESSING
    NUMBER_SLICES = 25
    K = 5 # number of representative colors by slice
    # MELODY GENERATION
    SAMPLE_RATE = 44100 # 44.1 KHz - standard used in most CDs and digital audio formats
    T = 0.2 # second duration
    FREQUENCIES = np.linspace(start=20, stop=2000, num=255) # range of frequencies to use
    OCTAVE_VALUES = [ # frequencies to use as octave values
        0.5, 1, 16/15, 9/8, 6/5, 5/4, 4/3, 45/32, 3/2, 8/5, 5/3, 9/5, 15/8, 2
    ]
    AMPLITUDE_FACTOR = 0.5
    # AUDIO POST-PROCESSING
    SOUND_EFFECTS = [ # Pedalboard effects/plug-ins, order matters
        Chorus(depth=0.15),
        Delay(delay_seconds=0.1),
        Reverb(room_size=0.5),
    ]


def rgb_to_octave(avg_rgb: int, octave_values: list) -> float:
    # generate equidistant thresholds for the RGB avg values
    thresholds = np.linspace(start=0, stop=255, num=len(octave_values))
    assert len(octave_values) == len(thresholds)

    octave = octave_values[0]

    for idx, threshold in enumerate(thresholds):
        if avg_rgb < threshold:
            octave = octave_values[idx]
            break

    return octave


def image_to_melody(image: np.ndarray):
    rgb_img = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)

    all_representative_pixels = get_representative_pixels(
        rgb_img, Conf.NUMBER_SLICES, Conf.K
    )

    df_repixels = pd.DataFrame(
        all_representative_pixels, columns=["red", "green", "blue"]
    )
    print(f"Number of representative pixels: {df_repixels.shape[0]}")

    df_repixels["avg_rgb"] = df_repixels.apply(
        lambda row : int(row["red"] + row["green"] + row["blue"]) // 3,
        axis=1,
    )

    # Add new column with the mapped frequency of the pixel values
    df_repixels["notes"] = df_repixels.apply(
        lambda row : Conf.FREQUENCIES[int(row["avg_rgb"])], axis=1
    )

    # Add column with the mapped octave value using the saturation value
    df_repixels["octave"] = df_repixels.apply(
        lambda row : rgb_to_octave(int(row["avg_rgb"]), Conf.OCTAVE_VALUES), axis=1
    )  

    print("STATS", "-"*21)
    print(df_repixels.describe())
    print("-"*21)

    return build_playable_audio(
        df_repixels,
        np.sin,
        sample_rate=Conf.SAMPLE_RATE,
        time=Conf.T,
        amplitude_factor=Conf.AMPLITUDE_FACTOR,
    )
