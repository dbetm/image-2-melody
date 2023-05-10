import argparse
import importlib
import os
from types import ModuleType

import cv2
import numpy as np
from scipy.io import wavfile

from image_to_melody import audio_processor, video_maker as vid_mk


IMAGES_PATH = "sample_images/"
TEMPLATE_OUTPUT_PATH = "outputs/exp_{experiment_id}/{image_name}/"
SAMPLE_RATE = 44100 
IMAGE_FORMATS = {"png", "jpg", "jpeg"}


def create_and_save_video(
    img: np.ndarray, base_output_path: str, audio_path: str, n_slices: int
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
        n_slices=n_slices,
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


def load_module(id: int) -> ModuleType:
    return importlib.import_module(
        name=f"experiments.exp_{id}.exp_{id}"
    )


def run_exp(exp: ModuleType, create_video: bool = False):
    """This function runs a pipeline for the given experiment."""
    images_filenames = os.listdir(path=IMAGES_PATH)
    images_filenames.sort()
    images_filenames = ["008_atardecer.jpg"]

    for img_filename in images_filenames:
        img_name, file_extension = img_filename.split(".")

        if file_extension not in IMAGE_FORMATS:
            continue

        print(img_filename)

        base_output_path = TEMPLATE_OUTPUT_PATH.format(
            experiment_id=exp.Conf.EXPERIMENT_ID,
            image_name=img_name
        )

        if not os.path.exists(base_output_path):
            os.makedirs(base_output_path)

        img_full_path = os.path.join(IMAGES_PATH, img_filename)
        img = cv2.imread(filename=img_full_path)
        audio_filename = img_filename.split(".")[0]
        audio_filename += ".wav"

        audio = exp.image_to_melody(img)
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
            effects=exp.Conf.SOUND_EFFECTS,
        )

        if create_video:
            create_and_save_video(
                img=img,
                base_output_path=base_output_path,
                audio_path=effected_audio_path,
                n_slices=exp.Conf.NUMBER_SLICES,
            )

        print("-"*42)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp", type=int, required=False, help="Id of the experiment to run"
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Create a video or not"
    )
    args = parser.parse_args()
    
    exp = load_module(args.exp)

    print(f"Executing exp_{args.exp}")
    print(exp.__doc__)

    run_exp(exp=exp, create_video=args.video)