import librosa
import numpy as np
from numba import boolean
from scipy.io import wavfile

import constants

def load_audio_from_path(path: str, samplerate: int, mono : boolean=True):
    audio, sr = librosa.load(path, sr=samplerate, mono=mono, dtype=np.float32)
    # Madmon handles the samplerate directly in the Signal class, but let's stick to Librosa for now
    # audio = madmom.audio.signal.Signal(path, mono=mono, dtype=np.float32)
    return audio


# Using scipy.wavfile because soundfile is clipping if not normalized to (-1, 1) but the gain is lower when normalized
def write_audio_to_path(path: str, audio_buffer: np.ndarray, samplerate: int):
    if len(audio_buffer.shape) > 1:
        print('WRITING - transposing')
        audio_buffer = audio_buffer.transpose()
    wavfile.write(path, samplerate, audio_buffer)


# Return the similarity of two audio buffer (number of occurrences that match) between 0 and 1
# Todo : add an epsilon tolerance to use this
def compare_mono_buffer(buffer_a, buffer_b) -> float:
    match = sum(a == b for a, b in zip(buffer_a, buffer_b))
    return (match / len(buffer_b))

# Slice an audio file at onset and create new ones
def slice(input_audio_buffer):
    slicer = Slicer(sr=constants.SAMPLERATE)
    for audio in slicer.slice(waveform=input_audio_buffer):
        print(audio)
        #render_wav(audio)


if __name__ == "__main__":
    print('nothing to run in this file yet')
