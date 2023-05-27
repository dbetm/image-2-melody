"""The image is read in RGB and HSV color spaces. In the RGB color space, representative pixels are extracted to select notes from chords. In the HSV color space, the H value is averaged from a 15% sample of pixels to select a group of MIDI instruments that are mapped to a color-emotion."""
from os import getenv

import cv2
import numpy as np
import pandas as pd
from mido import Message, MidiFile, MidiTrack, MetaMessage


from experiments.exp_2.instrument_groups import *
from image_to_melody.audio_processor import synthesize
from image_to_melody.connections import download_sound_font
from image_to_melody.img_utils import (
    get_pixel_median_values, get_representative_pixels
)


class Conf:
    # General
    EXPERIMENT_ID = 2
    # IMAGE PROCESSING
    NUMBER_SLICES = 25
    K = 4 # number representative pixels to get from each slice
    # MELODY GENERATION
    SAMPLE_RATE = 44100 # 44.1 KHz - standard used in most CDs and digital audio format
    NOTE_DURATION = 500
    # AUDIO POST-PROCESSING
    TEMP_MIDI_FILEPATH = "experiments/tmp/melody.mid"
    TEMP_AUDIO_FILEPATH = "experiments/tmp/melody.wav"
    SOUND_EFFECTS = [] # Pedalboard effects
    SOUND_FONT_FILEPATH = "experiments/tmp/sound_font.sf2"
    # VIDEO GENERATION
    RATE_IMG_REPETITION = 2
    FPS = 3


class Track(MidiTrack):
    def __init__(self, instrument: Instrument, channel: int, note_duration: int = 480):
        self.instrument = instrument
        self.channel = channel
        self.thresholds = np.linspace(start=0, stop=255, num=len(instrument.notes))
        self.note_duration = note_duration

        super().__init__()

        self.append(
            Message(
                type="program_change",
                program=instrument.midi_num,
                channel=self.channel,
                time=0,
            )
        )

    def add_note(self, pixel_channel_value: int):
        note = self.instrument.notes[0]

        for idx, threshold in enumerate(self.thresholds):
            if pixel_channel_value < threshold:
                note = self.instrument.notes[idx]
                break

        self.append(
            Message(
                type="note_on",
                channel=self.channel,
                note=note,
                velocity=self.instrument.volume,
                time=0,
            )
        )
        self.append(
            Message(
                type="note_off",
                channel=self.channel,
                note=note,
                velocity=self.instrument.volume,
                time=self.note_duration,
            )
        )

    def finish(self):
        self.append(MetaMessage(type="end_of_track", time=0))


def select_instrument_group(image: np.ndarray) -> dict:
    hue_median, _, _ = get_pixel_median_values(image)

    print(">> hue median", hue_median)

    if hue_median < 30.0:
        return RED_GROUP
    elif hue_median >= 30.0 and hue_median < 60.0:
        return YELLOW_GROUP
    elif hue_median >= 60.0 and hue_median < 94.0:
        return GREEN_GROUP
    elif hue_median >= 94.0 and hue_median < 100.0:
        return BLUE_2_GROUP
    elif hue_median >= 100.0 and hue_median < 142.0:
        return BLUE_GROUP
    elif hue_median >= 142.0 and hue_median < 155.0:
        return PURPLE_GROUP
    elif hue_median >= 155.0 and hue_median < 165.0:
        return PINK_GROUP

    return RED_2_GROUP


def get_pixel_rgb_value(row: dict, instrument: Instrument) -> int:
    if instrument.color_channel == "avg_all":
        return (row["red"] + row["green"] + row["blue"]) // 3

    return row[instrument.color_channel]


def image_to_melody(image: np.ndarray):
    rgb_img = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2HSV)

    all_representative_pixels = get_representative_pixels(
        rgb_img, Conf.NUMBER_SLICES, Conf.K
    )

    df_repixels = pd.DataFrame(
        all_representative_pixels, columns=["red", "green", "blue"]
    )
    instrument_group = select_instrument_group(hsv_img)

    if not getenv("PROD", None):
        print(instrument_group["description"])

    instruments = instrument_group["instruments"]

    # Generate melody
    mid = MidiFile()

    for idx, instrument in enumerate(instruments):
        track = Track(
            instrument=instrument, channel=idx, note_duration=Conf.NOTE_DURATION
        )
        mid.tracks.append(track)

    for idx, row in df_repixels.iterrows():
        for track in mid.tracks:
            track.add_note(
                pixel_channel_value=get_pixel_rgb_value(row, track.instrument)
            )

    # mark each track as finished
    for track in mid.tracks:
        track.finish()

    # Synthesize melody
    mid.save(Conf.TEMP_MIDI_FILEPATH)

    download_sound_font(Conf.SOUND_FONT_FILEPATH)
    synthesize(
        midi_file_path=Conf.TEMP_MIDI_FILEPATH,
        output_path=Conf.TEMP_AUDIO_FILEPATH,
        sample_rate=Conf.SAMPLE_RATE,
        sound_font_filepath=Conf.SOUND_FONT_FILEPATH,
    )

    return Conf.TEMP_AUDIO_FILEPATH
