import logging as logger
from datetime import datetime
from os import environ, path, remove

import streamlit as st

from run_exp import load_module, run_exp


logger.basicConfig(level=logger.INFO)

IMAGE_KEY = "image"
EXP_ID_KEY = "expID"
ALLOWED_EXP_IDS = [2, 3]


def save(uploaded_file, path: str):
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())


def reset():
    if st.session_state.get(IMAGE_KEY):
        del st.session_state[IMAGE_KEY]


def remove_tmp_files(filepaths: list):
    for filepath in filepaths:
        try:
            remove(filepath)
            logger.info(f"Deleted file: {filepath}")
        except OSError as e:
            logger.warning(f"Error deleting file: {filepath} - {e}")


def resolve_exp(exp_id: int) -> tuple:
    assert exp_id in ALLOWED_EXP_IDS, f"Select an appropiate experiment id {ALLOWED_EXP_IDS}"

    exp_module = load_module(exp_id)
    exp_info = exp_module.__doc__

    return (exp_module, exp_info)


def run():
    st.title("Image to Melody")
    st.write("Upload an image and we'll convert it into a short musical video.")

    selected_exp_id = st.selectbox(
        "Choose an experiment", ALLOWED_EXP_IDS, on_change=reset
    )
    exp_module, exp_info = resolve_exp(selected_exp_id)

    st.write("Experiment info:", exp_info)

    uploaded_file = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"], key=IMAGE_KEY
    )

    if uploaded_file is not None:
        image_filename = uploaded_file.name
        image_filepath = path.join("sample_images", image_filename)

        save(uploaded_file, path=image_filepath)

        st.image(uploaded_file, width=300)

        logger.info({"image": image_filename})

        with st.spinner(f"Running experiment {selected_exp_id} for image: {image_filename}"):
            video_filepath = run_exp(
                exp=exp_module, image_filenames=[image_filename], create_video=True
            )

        st.success("Video generated successfully!")
        st.write("Download your video:")

        now_str = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"image_to_melody_{now_str}.mp4"

        with open(video_filepath, "rb") as f:
            logger.info({"video": filename})
            st.download_button("Download", f, file_name=filename, on_click=reset)

        audio_filepath = video_filepath.replace(
            "final_audio.mp4", image_filename.rsplit(".", 1)[0] + ".wav"
        )
        remove_tmp_files([video_filepath, image_filepath, audio_filepath])



if __name__ == "__main__":
    environ["PROD"] = "1"
    run()