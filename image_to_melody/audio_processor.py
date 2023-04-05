from typing import Any, Callable, List

import numpy as np
import pandas as pd
from pedalboard import Pedalboard
from pedalboard.io import AudioFile


# MUSICAL SCALES - FREQUENCIES
A_NATURAL_MINOR =  [220.00, 233.08, 261.63, 293.66, 311.13, 349.23, 392.00]
A_HARMONIC_MINOR = [220.00, 246.94, 261.63, 293.66, 329.63, 349.23, 415.30]
A_MAJOR =          [220.00, 246.94, 277.18, 293.66, 329.63, 370.00, 415.30]
A_HARMONIC_MAJOR = [220.00, 246.94, 261.63, 293.66, 329.63, 366.67, 440.00]


def build_playable_audio(
    df: pd.DataFrame,
    gen_singal_fn: Callable,
    sample_rate: int,
    time: float,
    amplitude_factor: float = 1.0
) -> np.ndarray:
    """Given a dataframe with frequencies notes and octaves, create an audio
    using the sample rate and time variable duration `time`."""
    frequencies = df["notes"].to_numpy()
    octaves = df["octave"].to_numpy()

    # t represents an array of int(T*sample_rate) time values starting from 0 
    # and ending at T, with a fixed duration between each sample
    t = np.linspace(0, time, int(time*sample_rate), endpoint=False)

    song = np.array([])

    for octave, freq, in zip(octaves, frequencies):
        val = freq * octave
        # Represent each note as a sign wave
        note  = amplitude_factor * gen_singal_fn(2*np.pi*val*t)
        song  = np.concatenate([song, note]) # Add notes into song array to make song

    return song


def improve_audio(audio_path: str, effects: List[Any]) -> str:
    """Apply a highpass filter and a compressor effect to improve the sound."""
    # Make a Pedalboard object, containing multiple audio plugins:
    board = Pedalboard(plugins=[effect for effect in effects])

    with AudioFile(audio_path) as f:
        new_path = audio_path.replace(".wav", "_effected.wav")
        with AudioFile(new_path, "w", f.samplerate, f.num_channels) as o:
            # Read one second of audio at a time, until the file is empty:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))

                # Run the audio through our pedalboard:
                effected = board(chunk, f.samplerate, reset=False)

                # Write the output to our output file:
                o.write(effected)

    return new_path