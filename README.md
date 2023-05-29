# Image to Melody

This repository project contains several experiments with different approaches to **generate melodies from images**.

In each one, the image is divided in vertical slices where representative pixels are obtained. With those pixels notes are generated, how? (that's the experimental part).

The basic idea is that in the light the frequency determines the color (so it's captured in a photo) and in the sounds the frequency determines the tone/pitch.

When running a experiment you can generate a video too, like [this - Northern Lights](https://youtu.be/fh1Ca0vpPEI) which is generated using the experiment #2 or [this - The Starry Night](https://youtu.be/2mMM9h8iYG4) generated using a trained neural network (experiment #3).

Alternatively, you can use the web [Streamlit app](https://dbetm-image-2-melody-web-app-eydksr.streamlit.app/).


## Setup

**Requeriments**

- [virtualenv](https://virtualenv.pypa.io/en/latest/)

**Create virtual environment**

`virtualenv .venv --python=python3.8`

**Activate virtual environment**

`source .venv/bin/activate`

**Install base dependencies**

`pip install -r requirements-base.txt`


## Experiments

| id | description                                                                                                                                                                                                                                                                                   | recommended for                                  |
|----|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------|
| 0  | HSV values from pixels are linearly mapped to frequencies selected from musical scales. Then using those frequencies and a sinusoidal wave the sound is generated.                                                                                                                            | nature photos (sunset, clouds, flowers)          |
| 1  | The intensity of RGB values are averaged and then using those averages frequencies are selected in a linear way from a wide range of frequencies (16 Hz - 200 Khz). Apply Chorus, Delay y Reverb effects.                                                                                     | astronomy pictures                               |
| 2  | The image is read in RGB and HSV color spaces. In the RGB color space, representative pixels are extracted to select notes from chords. In the HSV color space, the H value is averaged from a 15% sample of pixels to select a group of MIDI instruments that are mapped to a color-emotion. | Photos with natural light, no much solid colors. |
| 3  | A trained Long-Short Term Memory RNN model is used to generate the next K notes of a melody given a sequence of them mapped linearly from Hue values of pixels. The music is based on the 6 suites of cello written by J. S. Bach.                                                            | Any type of photo.                               |


## Run an experiment

1) Install dependencies for a specific experiment.

`invoke setup-experiment --exp {id}`

Where `id` is the identifier of the experiment. 

**Important note:** If you wanna run the experiment 2 or 3, you need to install `fluidsynth` too at the OS level which helps us to synthesize the audios from MIDI notes, in Linux Ubuntu you can install with: `apt-get install fluidsynth`

2) Paste your image or images inside of: [sample_images/](sample_images/)

2) Then run the Python script, it will process each image:

```python
python3 run_exp.py --exp {id} --video
```

If you only want to generate the wav audio, then don't pass the `--video` flag.
And find the results on outputs

---------------------

## Contribute

This is a just-for-fun project, so feel free to contribute:
- Fork the repository.
- Create a new branch from the `main` branch.
- Push your branch and open a Pull Request targeting this repository.

Or maybe, if you detect a bug or something wrong can open an issue.