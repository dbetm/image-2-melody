import os
from http import HTTPStatus

import requests


SOUND_FONT_REMOTE_FILE_URL = (
    "https://github.com/ibireme/SF2Piano/raw/master/TestSF2/GeneralUser%20GS%20SoftSynth%20v1.44.sf2"
)



def download_sound_font(dest_local_filepath: str):
    """Download sound font file, which is necessary to synthesize MIDI audio files."""
    if os.path.isfile(dest_local_filepath):
        print("Skipping download the sound font file. It already exists.")
        return
    
    response = requests.get(url=SOUND_FONT_REMOTE_FILE_URL)

    if response.status_code == HTTPStatus.OK:
        with open(dest_local_filepath, "wb") as f:
            f.write(response.content)
            print(f"Successfully downloaded sound fond file at: {dest_local_filepath}")
    else:
        raise Exception(
            "Error trying to download the sound font file. Which is required to"
            " synthesize MIDI files."
        )