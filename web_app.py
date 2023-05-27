import logging as logger
from datetime import datetime
from os import environ

import streamlit as st

from experiments.exp_2 import exp_2
from run_exp import run_exp


logger.basicConfig(level=logger.INFO)

IMAGE_KEY = "image"


def save(uploaded_file, path: str):
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())


def reset():
    del st.session_state[IMAGE_KEY]


def run():
    st.title("Image to Melody")
    st.write("Upload an image and we'll convert it into a short musical video.")

    uploaded_file = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"], key=IMAGE_KEY
    )

    if uploaded_file is not None:
        image_filename = uploaded_file.name

        save(uploaded_file, path=f"sample_images/{image_filename}")

        st.image(uploaded_file, width=300)

        logger.info({"image": image_filename})

        with st.spinner(f"Running experiment 2 for image: {image_filename}"):
            video_filepath = run_exp(
                exp=exp_2, image_filenames=[image_filename], create_video=True
            )

        st.success("Video generated successfully!")
        st.write("Download your video:")

        now_str = datetime.now().strftime("%Y%m%d_%H%m")
        filename = f"{now_str}_image_to_melody.mp4"

        with open(video_filepath, "rb") as f:
            logger.info({"video": filename})
            st.download_button("Download", f, file_name=filename, on_click=reset)



if __name__ == "__main__":
    environ["PROD"] = "1"
    run()