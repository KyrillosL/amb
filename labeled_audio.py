import os
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np

from loguru import logger

import constants
import preprocess
from audio_utils import load_audio_from_path, write_audio_to_path
from midi_file_generator import MidiFileGenerator
from musical import get_pitch
from temporal import detect_onset_librosa, get_tempo, get_note_duration_for_onset, cut_audio_files, get_rms
from temporal import get_onset_envelope, get_onset_envelope_tempo, get_volume


def get_freq_at_onset_sample(onset_sample, freq_index_second, freq):
    onset_second = librosa.core.samples_to_time(onset_sample, sr=constants.SAMPLERATE)
    closest_index = min(range(len(freq_index_second)), key=lambda i: abs(freq_index_second[i] - onset_second))

    # logger.debug(freq.shape)
    # Todo : throw an error if out of bounds
    freq_value_at_time = freq[closest_index]
    logger.debug('onset sample ' + str(onset_sample) + ' onset second ' + str(onset_second) + ' closest_index ' + str(
        closest_index) + ' freq_value_at_time ' + str(freq_value_at_time))
    return freq_value_at_time


class Note:
    def __init__(self, start_sample, duration_sample, freq, volume):
        self.start_sample = start_sample
        self.start_second = librosa.core.samples_to_time(start_sample, sr=constants.SAMPLERATE)
        self.duration_sample = duration_sample
        self.duration_second = librosa.core.samples_to_time(duration_sample, sr=constants.SAMPLERATE)
        self.freq = freq
        self.midi_note_number = int(librosa.hz_to_midi(freq)) + 12
        self.volume = volume

    def __repr__(self):
        return 'Start ' + str(self.start_second), ' duration ' + str(self.duration_second) + ' note ' + str(
            self.midi_note_number) + ' volume ' + str(self.volume)


class LabeledAudio:
    def __init__(self, file: Path):
        logger.info('Creating track for ' + file.name)
        self.path = file
        self.name = file.name

        self.notes: list[Note] = []

        # Audio Loading
        self.audio_buffer = load_audio_from_path(file, constants.SAMPLERATE)
        self.rms = get_rms(audio_buffer=self.audio_buffer)
        self.min_rms = np.min(self.rms)
        self.max_rms = np.max(self.rms)
        self.onset_envelope = get_onset_envelope(self.audio_buffer)

        times = librosa.times_like(self.onset_envelope, sr=constants.SAMPLERATE)
        plt.figure()
        plt.plot(times, self.onset_envelope, label='Onset strength')
        # plt.show()

        self.onset_envelope_tempo = get_onset_envelope_tempo(self.audio_buffer)
        self.tempo = get_tempo(self.onset_envelope_tempo)

        # Todo move all preprocessing to the preprocess file and render one file (not only filtered)

        # Used remove some high frequencies so crepe can focus on lower ones as we work with bass
        self.path_filtered = os.path.join('filtered', self.name)
        if not os.path.exists(self.path_filtered):
            filtered_audio = preprocess.filter_audio(audio_buffer=self.audio_buffer)
            # Write the file on disk so we don't have to redo it each time
            write_audio_to_path(path=self.path_filtered, audio_buffer=filtered_audio, samplerate=constants.SAMPLERATE)

        self.extract()

    def __repr__(self):
        out = self.name + ' tempo ' + str(self.tempo) + '\n'
        for note in self.notes:
            out = out + '\t ' + str(note.__repr__()) + '\n'
        return out

    # Fill list of notes
    def extract(self):
        logger.info('Main extract', )
        # It's actually faster to reload the audio than creating a Tensor from a numpy.ndarrays
        # freq_index_second, frequencies = get_pitch(self.path_filtered)
        freq_index_second, frequencies = get_pitch(self.path)
        logger.info('index frequencies ' + str(freq_index_second))
        logger.info('frequencies ' + str(frequencies))
        # Get onsets using Librosa
        onset_samples = detect_onset_librosa(self.onset_envelope)

        # Onset Samples used to cut
        onset_samples_cut = detect_onset_librosa(self.onset_envelope_tempo)

        # Get note duration for each onset
        notes_duration_sample = get_note_duration_for_onset(onset_samples_cut, self.audio_buffer)

        for onset_s in onset_samples:
            closest_index_onset_tempo = min(range(len(onset_samples_cut)), key=lambda i: abs(onset_samples_cut[i] - onset_s))
            duration_sample = notes_duration_sample[onset_samples_cut[closest_index_onset_tempo]]
            volume = get_volume(onset_s, self.audio_buffer, duration_sample, self.min_rms, self.max_rms)
            freq = get_freq_at_onset_sample(onset_s, freq_index_second, frequencies)
            self.notes.append(Note(start_sample=onset_s, duration_sample=duration_sample, freq=freq, volume=volume))

        # Used for debug purposes
        cut_audio_files(onset_samples, self.audio_buffer, self.name, 'librosa')

    def generate_midi_file(self):
        mfg = MidiFileGenerator(self.name, self.tempo, self.notes)
        mfg.write(os.path.join('midi', self.name.replace('.wav', '.mid')))


if __name__ == "__main__":
    logger.error('Please run the main file')
