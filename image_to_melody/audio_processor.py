from functools import lru_cache
from typing import Any, Callable, List

import numpy as np
import pandas as pd
from pedalboard import Pedalboard
from pedalboard.io import AudioFile
from midi2audio import FluidSynth


# MUSICAL SCALES - FREQUENCIES
A_NATURAL_MINOR =  [220.00, 233.08, 261.63, 293.66, 311.13, 349.23, 392.00]
A_HARMONIC_MINOR = [220.00, 246.94, 261.63, 293.66, 329.63, 349.23, 415.30]
A_MAJOR =          [220.00, 246.94, 277.18, 293.66, 329.63, 370.00, 415.30]
A_HARMONIC_MAJOR = [220.00, 246.94, 261.63, 293.66, 329.63, 366.67, 440.00]


# CHORDS - MIDI FREQUENCIES
class MidiChords:
    # Major Chords
    A_MAJOR = [57, 61, 64] # A, C#, E
    B_MAJOR = [59, 63, 66] # B, D#, F#
    C_MAJOR = [60, 64, 67] # C, E, G 
    E_MAJOR = [64, 68, 71] # E, G#, B
    G_MAJOR = [55, 59, 62] # G, B, D
    F_MAJOR = [65, 69, 72] # F, A, C 
    # Minor Chords
    A_MINOR = [57, 60, 64] # A, C, E
    B_MINOR = [59, 62, 66] # B, D, F
    D_MINOR = [62, 65, 69] # D, F, A
    E_MINOR = [64, 67, 71] # E, G, B
    F_MINOR = [65, 58, 72] # F, #G, C
    # Root notes
    C_MAJOR_ROOT = [48]  # C
    F_MAJOR_ROOT = [53]  # F
    G_MAJOR_ROOT = [55]  # G


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
    if len(effects) == 0:
        return audio_path

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


@lru_cache(maxsize=10)
def group_notes(notes: list) -> list:
    """Grant that notes are unique, then sort them in ascending order."""
    notes = list(set(notes))
    notes.sort()

    return notes


def synthesize(
    midi_file_path: str, output_path: str, sample_rate: int, sound_font_filepath: str
) -> None:
    """Synthesize the midi file and save it into the output path as an audio file."""
    fs = FluidSynth(sound_font=sound_font_filepath, sample_rate=sample_rate)
    fs.midi_to_audio(midi_file_path, output_path)