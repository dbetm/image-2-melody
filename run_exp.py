import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile

from exp_hardcoding_frequencies import exp_0

BASE_OUTPUT_PATH = "output_music/exp_{experiment_id}/"
IMAGES_PATH = "sample_images/"
SAMPLE_RATE = 44100 


def run_exp_0():
    """This experiment generate audio from the image, then the video composed with audio.
    The approach is hardcode frequencies of a selected musical scale using the HUE value of
    the image. The musical scale is selected with a simple heuristic using the average RGB 
    values of the image from a sample.
    """
    images_filenames = os.listdir(path=IMAGES_PATH)
    images_filenames.sort()
    images_filenames = ["003_starry_night.jpg"]

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

        audio = exp_0.image_to_melody(img)

        base_output_path = BASE_OUTPUT_PATH.format(experiment_id=2)
        if not os.path.exists(base_output_path):
            os.makedirs(base_output_path)

        output_path = os.path.join(base_output_path, audio_filename)

        # save audio 
        wavfile.write( 
            output_path,
            rate=SAMPLE_RATE,
            data=audio.astype(np.float32)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp", type=int, required=False, default=0, help="Id of the experiment to run"
    )
    args = parser.parse_args()

    eval(f"run_exp_{args.exp}()")