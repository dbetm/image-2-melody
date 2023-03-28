from pedalboard import Pedalboard, Compressor, HighpassFilter
from pedalboard.io import AudioFile


def improve_audio(audio_path: str) -> str:
    """Apply a highpass filter and a compressor effect to improve the sound."""
    # Make a Pedalboard object, containing multiple audio plugins:
    board = Pedalboard([
        HighpassFilter(),
        Compressor(threshold_db=0, ratio=25),
    ])

    with AudioFile(audio_path) as f:
        new_path = audio_path.replace(".wav", "_effected.wav")
        with AudioFile(new_path, "w", f.samplerate, f.num_channels) as o:
            # Read one second of audio at a time, until the file is empty:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))

                # Run the audio through our pedalboard:
                effected = board(chunk, f.samplerate, reset=False)

                # Write the output to our output file:
                o.write(effected)

    return new_path


if __name__ == '__main__':
    path_ = "output_music/exp_0/010_galaxy.wav"

    improve_audio(path_)