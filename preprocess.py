import os
from scipy.signal import butter, lfilter
import numpy as np
import librosa
from loguru import logger
import numpy as np

import constants
import audio_utils

def remove_stereo_sides(audio_buffer):
    # Todo : check how librosa mixdown to mono
    return audio_buffer


def lp_filter(audio_buffer: np.ndarray) -> np.ndarray:
    logger.info('Filtering audio signal')
    b, a = butter(N=constants.ORDER, Wn=constants.CUTOFF, fs=constants.SAMPLERATE, btype='low', analog=False, output='ba')
    y = lfilter(b, a, audio_buffer)
    return y


def filter_audio(audio_buffer: np.ndarray) -> np.ndarray:
    return lp_filter(audio_buffer)


def resave_to_mono(input_folder: str, output_folder: str, name: str):
    logger.info('Resaving ' + name + ' to mono in folder ' + input_folder)
    input_path = os.path.join(input_folder, name)
    output_path = os.path.join(output_folder, name)
    audio, sr = librosa.load(input_path, sr=constants.SAMPLERATE, mono=True, dtype=np.float32)
    audio_utils.write_audio_to_path(output_path, audio, sr)

