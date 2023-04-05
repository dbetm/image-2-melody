import argparse
import os

import cv2
import numpy as np
from pedalboard import Compressor, HighpassFilter
from scipy.io import wavfile
from tqdm import tqdm

from exp_hardcoding_frequencies import exp_0, exp_1
from image_to_melody import audio_processor, video_maker as vid_mk


IMAGES_PATH = "sample_images/"
TEMPLATE_OUTPUT_PATH = "outputs/exp_{experiment_id}/{image_name}/"
SAMPLE_RATE = 44100 
IMAGE_FORMATS = {"png", "jpg", "jpeg"}


def create_and_save_video(
    img: np.ndarray, base_output_path: str, audio_path: str
) -> None:
    """Create dir to save the video, generate frames and then save the video with
    audio in the `base_output_path` given."""
    video_folder = os.path.join(
        base_output_path, "frames/"
    )

    if not os.path.isdir(video_folder):
        os.makedirs(video_folder)
    
    vid_mk.generate_fotograms(
        output_path=video_folder,
        img=img,
        n_slices=exp_0.NUMBER_SLICES,
    )

    tmp_video_path = os.path.join(base_output_path, "final.mp4")

    video_filepath = vid_mk.create_video(
        base_images_dir=video_folder,
        audio_path=audio_path,
        output_path=tmp_video_path,
    )

    # delete tmp video without audio
    os.remove(tmp_video_path)

    print(f"Video saved at: {video_filepath}")


def run_exp_0(create_video: bool = False):
    """This experiment generate audio from the image, then the video composed with audio.
    The approach is hardcode frequencies of a selected musical scale using the HUE value of
    the image. The musical scale is selected with a simple heuristic using the average RGB 
    values of the image from a sample.
    """
    images_filenames = os.listdir(path=IMAGES_PATH)
    images_filenames.sort()
    images_filenames = ["002_pilares_de_la_creacion.png"]

    for img_filename in tqdm(images_filenames):
        img_name, file_extension = img_filename.split(".")

        if file_extension not in IMAGE_FORMATS:
            continue

        print(img_filename)

        base_output_path = TEMPLATE_OUTPUT_PATH.format(
            experiment_id=exp_0.EXPERIMENT_ID,
            image_name=img_name
        )

        if not os.path.exists(base_output_path):
            os.makedirs(base_output_path)

        img_full_path = os.path.join(IMAGES_PATH, img_filename)
        img = cv2.imread(filename=img_full_path)
        audio_filename = img_filename.split(".")[0]
        audio_filename += ".wav"

        audio = exp_0.image_to_melody(img)
        audio_output_path = os.path.join(base_output_path, audio_filename)

        # save audio 
        wavfile.write( 
            audio_output_path,
            rate=SAMPLE_RATE,
            data=audio.astype(np.float32)
        )

        # improve audio
        effected_audio_path = audio_processor.improve_audio(
            audio_path=audio_output_path, 
            effects=[
                HighpassFilter(),
                Compressor(threshold_db=0, ratio=25),
            ]
        )

        if create_video:
            create_and_save_video(
                img=img, 
                base_output_path=base_output_path,
                audio_path=effected_audio_path
            )


def run_exp_1(create_video: bool = False):
    """The approach is to map RGB average values by pixel to a wide range of
    frequencies in a linear way, example 0 RGB average value is map to 16 Hz
    and so until 255 RVB average and 2 KHz."""
    images_filenames = os.listdir(path=IMAGES_PATH)
    images_filenames.sort()

    for img_filename in images_filenames:
        img_name, file_extension = img_filename.split(".")

        if file_extension not in IMAGE_FORMATS:
            continue

        print(img_filename)

        base_output_path = TEMPLATE_OUTPUT_PATH.format(
            experiment_id=exp_1.Conf.EXPERIMENT_ID,
            image_name=img_name
        )

        if not os.path.exists(base_output_path):
            os.makedirs(base_output_path)

        img_full_path = os.path.join(IMAGES_PATH, img_filename)
        img = cv2.imread(filename=img_full_path)
        audio_filename = img_filename.split(".")[0]
        audio_filename += ".wav"

        audio = exp_1.image_to_melody(img)
        audio_output_path = os.path.join(base_output_path, audio_filename)

        # save audio 
        wavfile.write( 
            audio_output_path,
            rate=SAMPLE_RATE,
            data=audio.astype(np.float32)
        )

        # improve audio
        effected_audio_path = audio_processor.improve_audio(
            audio_path=audio_output_path, 
            effects=exp_1.Conf.SOUND_EFFECTS,
        )

        if create_video:
            create_and_save_video(
                img=img, 
                base_output_path=base_output_path,
                audio_path=effected_audio_path
            )

        print("-"*42)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp", type=int, required=False, help="Id of the experiment to run"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=False,
        default="yes",
        choices=["yes, no"],
        help="Create video or not"
    )
    args = parser.parse_args()
    create_video_flag = args.video == "yes"

    eval(f"run_exp_{args.exp}({create_video_flag})")