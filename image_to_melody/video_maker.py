import os

import cv2
from moviepy import editor as mp
import numpy as np


DELTA_BRIGHT = 30
BASE_OUTPUT_VIDEOS = "output_videos/"


def highlight_pixel(channel_value: int, is_border: bool = False):
    if is_border:
        return 3 # 0 is completely black

    return min(255, channel_value + DELTA_BRIGHT)


def generate_fotograms(output_path: str, img: np.ndarray, n_slices: int):
    """Given a RGB image, create n_slices images highlighting the slice_i for
    each fotogram."""
    height, width, _ = img.shape
    fotogram_count = 1
    slice_width = width // n_slices
    percent_border_of_slice_width = 0.06 # 6%
    border_width = int(max(1, slice_width * percent_border_of_slice_width))
    print("border_width", border_width, "slice_width", slice_width)

    for slice_i in range(0, width, slice_width):
        # create a copy of the original image, then highlight the slice and 
        # save it inside the output path
        tmp_img = img.copy()

        for y in range(height):
            stop = min(slice_i + slice_width, width)

            for x in range(slice_i, stop):
                r, g, b = tmp_img[y][x]

                is_border = x >= (stop - border_width) or x < (slice_i + border_width)

                r = highlight_pixel(r, is_border)
                g = highlight_pixel(g, is_border)
                b = highlight_pixel(b, is_border)

                tmp_img[y][x] = (r, g, b)

        # save image
        fotogram_path = os.path.join(output_path, f"{fotogram_count}.png")
        cv2.imwrite(fotogram_path, tmp_img)

        fotogram_count += 1


def to_numeric(a: str):
    return int(a.split("/")[-1].split(".")[0])


def create_video(
    base_images_dir: str, audio_path: str, output_path: str, rate_img_repetition: int = 1
) -> str:
    """rate_img_repeat, indicates how many times a single image will be repeated and then
    added as a single frame."""
    audio = mp.AudioFileClip(filename=audio_path)
    # we want an image for each second
    fps = 1

    # get list of fotogram images
    img_files = [
        os.path.join(base_images_dir, f) for f in os.listdir(base_images_dir)
        if f.endswith(".png")
    ]
    img_files.sort(key=to_numeric)

    # Load the first image to get the size
    img = cv2.imread(img_files[0])
    height, width, _ = img.shape

    # create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Loop through the image files and add them to the video writer
    for image_file in img_files:
        img = cv2.imread(image_file)
        for _ in range(rate_img_repetition):
            video_writer.write(img)

    # Release the video writer
    video_writer.release()

    # Add the audio to the video file
    video = mp.VideoFileClip(output_path)
    video = video.set_audio(audio.set_duration(video.duration))
    # it's important to use another filename to make sure the process finishes correctly
    new_path = output_path.replace(".mp4","_audio.mp4")
    video.write_videofile(new_path, audio_codec="libmp3lame")

    return new_path


if __name__ == '__main__':
    filename = "003_starry_night"
    audio_path = f"outputs/exp_2/first_example.mp3"
    image_path = f"sample_images/{filename}.jpg"
    number_slices = 30
    video_folder = image_path.split("/")[1].split(".")[0]

    fotograms_output_path = os.path.join(
        "outputs/exp_2", "yupi"
    )

    if not os.path.isdir(fotograms_output_path):
        os.makedirs(fotograms_output_path)

    img = cv2.imread(filename=image_path)
    # color_img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)

    generate_fotograms(
        output_path=fotograms_output_path,
        img=img,
        n_slices=number_slices,
    )

    create_video(
        base_images_dir=fotograms_output_path,
        audio_path=audio_path,
        output_path=os.path.join(fotograms_output_path, "final.mp4")
    )
