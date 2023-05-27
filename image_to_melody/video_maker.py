import cv2
from moviepy import editor as mp
import numpy as np


DELTA_BRIGHT = 30
BASE_OUTPUT_VIDEOS = "output_videos/"


def highlight_pixel(channel_value: int, is_border: bool = False):
    if is_border:
        return 3 # 0 is completely black

    return min(255, channel_value + DELTA_BRIGHT)


def generate_fotograms(img: np.ndarray, n_slices: int, fps: int = 1):
    """Given a RGB image, create n_slices images highlighting the slice_i for
    each fotogram."""
    height, width, _ = img.shape
    fotogram_count = 1
    slice_width = width // n_slices
    percent_border_of_slice_width = 0.06 # 6%
    border_width = int(max(1, slice_width * percent_border_of_slice_width))
    border_width = min(border_width, 6)

    step = slice_width / fps
    step = slice_width if step < 1 else step
    print("border_width", border_width, "slice_width", slice_width, "step", step)

    for slice_i in np.arange(0, width, step):
        # create a copy of the original image, then highlight the slice and 
        # save it inside the output path
        tmp_img = img.copy()

        for y in range(height):
            stop = min(slice_i + slice_width, width)

            for x in np.arange(slice_i, stop, 1.0):
                x_int = int(x)
                r, g, b = tmp_img[y][x_int]

                is_border = x >= (stop - border_width) or x < (slice_i + border_width)

                r = highlight_pixel(r, is_border)
                g = highlight_pixel(g, is_border)
                b = highlight_pixel(b, is_border)

                tmp_img[y][x_int] = (r, g, b)

        fotogram_count += 1
        yield tmp_img


def to_numeric(a: str):
    return int(a.split("/")[-1].split(".")[0])


def create_video(
    img: np.ndarray,
    n_slices: int,
    audio_path: str,
    output_path: str,
    rate_img_repetition: int = 1,
    fps: int = 1, # deafult it's an image for each second
) -> str:
    """rate_img_repeat, indicates how many times a single image will be repeated and then
    added as a single frame."""
    audio = mp.AudioFileClip(filename=audio_path)

    height, width, _ = img.shape

    # create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Loop through the image files and add them to the video writer
    for image in generate_fotograms(img, n_slices, fps):
        for _ in range(rate_img_repetition):
            video_writer.write(image)

    # Release the video writer
    video_writer.release()

    # Add the audio to the video file
    video = mp.VideoFileClip(output_path)
    # video = video.set_audio(audio.set_duration(video.duration))
    video = video.set_audio(audio)
    # it's important to use another filename to make sure the process finishes correctly
    new_path = output_path.replace(".mp4","_audio.mp4")
    video.write_videofile(new_path, audio_codec="libmp3lame")

    return new_path
