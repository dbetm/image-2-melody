import argparse
import importlib
import os
import shutil
from types import ModuleType
from typing import List

import cv2
import numpy as np
from scipy.io import wavfile

from image_to_melody import audio_processor, video_maker as vid_mk
from image_to_melody.img_utils import resize_image


IMAGES_PATH = "sample_images/"
TEMPLATE_OUTPUT_PATH = "outputs/exp_{experiment_id}/{image_name}/"
SAMPLE_RATE = 44100 
IMAGE_FORMATS = {"png", "jpg", "jpeg"}
THRESHOLD_IMAGE_DIM = 1280


def create_and_save_video(
    img: np.ndarray,
    base_output_path: str,
    audio_path: str,
    n_slices: int,
    num_repetitions: int,
    fps: int,
) -> str:
    """Create dir to save the video, generate frames and then save the video with
    audio in the `base_output_path` given."""
    tmp_video_path = os.path.join(base_output_path, "final.mp4")
    assert fps > 0

    video_filepath = vid_mk.create_video(
        img=img,
        n_slices=n_slices,
        audio_path=audio_path,
        output_path=tmp_video_path,
        rate_img_repetition=num_repetitions,
        fps=fps,
    )

    # delete tmp video without audio
    os.remove(tmp_video_path)

    print(f"Video saved at: {video_filepath}")

    return video_filepath


def load_module(id: int) -> ModuleType:
    return importlib.import_module(
        name=f"experiments.exp_{id}.exp_{id}"
    )


def run_exp(
    exp: ModuleType,
    image_filenames: List[str],
    create_video: bool = False,
    fps: int = None,
):
    """This function runs a pipeline for the given experiment."""
    for img_filename in image_filenames:
        img_name, file_extension = img_filename.split(".")

        if file_extension not in IMAGE_FORMATS:
            continue

        print(">>", img_filename)

        base_output_path = TEMPLATE_OUTPUT_PATH.format(
            experiment_id=exp.Conf.EXPERIMENT_ID,
            image_name=img_name
        )

        if not os.path.exists(base_output_path):
            os.makedirs(base_output_path)

        img_full_path = os.path.join(IMAGES_PATH, img_filename)
        img = cv2.imread(filename=img_full_path)
        resized_img = resize_image(img, THRESHOLD_IMAGE_DIM)

        audio_filename = img_filename.split(".")[0]
        audio_filename += ".wav"

        audio = exp.image_to_melody(resized_img)
        audio_output_path = os.path.join(base_output_path, audio_filename)

        if isinstance(audio, np.ndarray):
            # save audio 
            wavfile.write( 
                audio_output_path,
                rate=SAMPLE_RATE,
                data=audio.astype(np.float32)
            )
        else: # audio is the path of the saved file
            shutil.copyfile(src=audio, dst=audio_output_path)

        # improve audio
        effected_audio_path = audio_processor.improve_audio(
            audio_path=audio_output_path, 
            effects=exp.Conf.SOUND_EFFECTS,
        )

        if create_video:
            return create_and_save_video(
                img=resized_img,
                base_output_path=base_output_path,
                audio_path=effected_audio_path,
                n_slices=exp.Conf.NUMBER_SLICES,
                num_repetitions=exp.Conf.RATE_IMG_REPETITION,
                fps=fps if fps else exp.Conf.FPS,
            )

        print("-"*42)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp", type=int, required=True, help="Id of the experiment to run"
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Create a video or not"
    )
    parser.add_argument(
        "--fps",
        type=int,
        required=False,
        help="Number of frames per second. A larger number generates a smoother video"
    )
    args = parser.parse_args()
    exp = load_module(args.exp)

    print(f"Executing exp_{args.exp}")
    print(exp.__doc__)

    images_filenames = os.listdir(path=IMAGES_PATH)
    images_filenames.sort()
    #images_filenames = ["001_hubble_deep_space.jpg"]

    run_exp(
        exp=exp,
        image_filenames=images_filenames,
        create_video=args.video,
        fps=args.fps
    )