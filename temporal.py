import os

import librosa
import madmom
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

import constants
from audio_utils import write_audio_to_path


# @apply_pad
def get_onset_envelope(audio_buffer):
    logger.info('Getting Onset Envelope')
    return librosa.onset.onset_strength(y=audio_buffer, sr=constants.SAMPLERATE)


def get_onset_envelope_tempo(audio_buffer):
    logger.info('Getting Onset Envelope')
    return librosa.onset.onset_strength(y=audio_buffer, sr=constants.SAMPLERATE, aggregate=np.median, n_mels=256)

def detect_onset_librosa(onset_envelope):
    logger.info('Detecting Onset via Librosa')
    times = librosa.times_like(onset_envelope, sr=constants.SAMPLERATE)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_envelope, sr=constants.SAMPLERATE, backtrack=True)
    onset_samples = librosa.core.frames_to_samples(onset_frames)

    plt.figure()
    plt.plot(times, onset_envelope, label='Onset strength')
    plt.vlines(times[onset_frames], 0, onset_envelope.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
    # plt.show()

    return onset_samples


def get_tempo(onset_envelope):
    tempo = librosa.feature.tempo(sr=constants.SAMPLERATE,
                                  onset_envelope=onset_envelope)  # Todo : try with madmom  # from madmom.features.beats import RNNBeatProcessor  # proc = TempoEstimationProcessor(fps=100)  # act = RNNBeatProcessor()('tests/data/audio/sample.wav')
    logger.info('Detected tempo ' + str(tempo))
    return tempo


def get_note_duration_for_onset(onset_samples, audio_buffer):
    logger.info('getting note duration for onset')

    durations_onsets = {}
    for index, sample in enumerate(onset_samples):  # [:-1]

        end = len(audio_buffer)
        if index != len(onset_samples) - 1:
            end = onset_samples[index + 1]
        buffer_chunk = audio_buffer[sample:end]

        rms = get_rms(buffer_chunk).squeeze(0)
        ratio_rms = round(len(buffer_chunk) / constants.LIBROSA_DEFAULT_HOP_LENGTH_ONSET) + 1

        """
        slicer = Slicer(sr=constants.SAMPLERATE, threshold=constants.THRESHOLD_SILENCE, min_length=301)
        sliced = slicer.slice(waveform=buffer_chunk)
        if len(sliced) > 0:
            sliced = np.concatenate(sliced)

        #print(sliced_concatenated)
        end_sample = len(sliced)
        durations_onsets[sample] = end_sample
        logger.debug('index ' + str(index) + ' onset start sample ' + str(sample) + ' duration time sample ' + str(end_sample))
        """

        threshold = 10 ** (constants.THRESHOLD_SILENCE / 20.)

        durations_onsets[sample] = len(buffer_chunk)
        previous_rms = rms[0]

        wait_for_frame = 2
        cut = False
        for i, rms_value in enumerate(rms):
            # First, look for increasing value. Once decreasing, search for the threshold
            if rms_value > previous_rms or i < wait_for_frame:
                continue

            if rms_value < threshold:
                cut = True
                # Silence detected
                time_cut_sample = ratio_rms * i
                logger.debug(
                    'index ' + str(index) + ' onset start sample ' + str(sample) + ' duration time sample ' + str(
                        time_cut_sample))
                durations_onsets[sample] = time_cut_sample

                break
        if not cut:
            logger.debug('index ' + str(index) + ' onset start sample ' + str(sample) + ' duration time sample ' + str(
                len(buffer_chunk)))

    return durations_onsets


def get_rms(audio_buffer):
    s = librosa.magphase(librosa.stft(audio_buffer, window=np.ones, center=True))[0]
    rms = librosa.feature.rms(S=s)

    fig, ax = plt.subplots(nrows=2, sharex=True)
    times = librosa.times_like(rms)
    ax[0].semilogy(times, rms[0], label='RMS Energy')
    ax[0].set(xticks=[])
    ax[0].legend()
    ax[0].label_outer()
    librosa.display.specshow(librosa.amplitude_to_db(s, ref=np.max), y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set(title='log Power spectrogram')

    # plt.show()
    return rms


# Hack : Pad audio buffer so we detect the first onset
# Not used anymore,
def apply_pad(fn_onset):
    def function(*audio):
        logger.info('Applying pad')
        # TODO : FINISH - Count number of 0 to deduce from pad - not working
        trimmed_audio = np.trim_zeros(*audio, 'f')
        nb_trimmed_values = len(*audio) - len(trimmed_audio)
        if nb_trimmed_values > constants.PAD_SIZE:
            print("COLD START - No need to pad - no cold start")
            return fn_onset(*audio)

        print("COLD START - Padding the audio buffer to avoid cold start")
        print("COLD START - nb_trimmed_values ", nb_trimmed_values)
        nb_zeros = constants.PAD_SIZE - nb_trimmed_values
        pad_array = np.zeros(nb_zeros)
        audio_padded = np.concatenate((pad_array, *audio))
        onset_frames = fn_onset(audio_padded)
        logger.debug('onset samples size ' + str(len(onset_frames)))
        nb_frame_padded = round(nb_zeros / constants.LIBROSA_DEFAULT_HOP_LENGTH_ONSET) + 1

        logger.debug('nb_frame_padded ' + str(nb_frame_padded))
        removed = []
        for o in onset_frames:
            """
            if o < nb_frame_padded:
                o = 0
            else:
                o -= nb_frame_padded
            removed.append(o)
            """
        removed = np.asarray(removed)
        return removed  # [o - nb_zeros for o in onset_samples]

    return function


# TODO : WIP
# @apply_pad
def get_onset_madmon(audio_buffer):
    # Display spec
    hop_size = madmom.audio.signal.HOP_SIZE
    spec = madmom.audio.spectrogram.Spectrogram(audio_buffer)
    plt.figure()
    plt.imshow(spec[:, :200].T, origin='lower', aspect='auto')

    # Using the SuperFlux Algo
    s = madmom.audio.signal.Signal(audio_buffer)  # , samplerate=constants.SAMPLERATE doesn't work
    s.sample_rate = constants.SAMPLERATE

    fs = madmom.audio.signal.FramedSignal(s)
    stft = madmom.audio.stft.STFT(fs)
    log_filt_spec = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(stft, num_bands=24)
    superflux = madmom.features.onsets.superflux(log_filt_spec)
    onset_frames = librosa.onset.onset_detect(onset_envelope=superflux, sr=constants.SAMPLERATE, backtrack=False)
    onsets_samples = librosa.core.frames_to_samples(onset_frames, hop_length=hop_size)

    plt.figure()
    plt.plot(superflux / superflux.max(), 'k:')
    # plt.show()

    # sodf = madmom.features.onsets.SpectralOnsetProcessor(onset_method='superflux', fps=200, filterbank = madmom.audio.filters.LogarithmicFilterbank, num_bands = 24, log = np.log10)
    # test = sodf(s)
    # onset_time2 = proc(test)
    # onsets_samples2 = [int(o * constants.SAMPLERATE) for o in onset_time2]

    # TODO : Try with note
    # proc = NotePeakPickingProcessor(fps=100)
    # act = RNNPianoNoteProcessor()('tests/data/audio/stereo_sample.wav')

    proc = madmom.features.onsets.OnsetPeakPickingProcessor(samplerate=constants.SAMPLERATE)
    # act = madmom.features.onsets.RNNOnsetProcessor()(s)
    # acts = madmom.features.onsets.SpectralOnsetProcessor()(s)
    act = madmom.features.onsets.CNNOnsetProcessor()(s)
    onset_time = proc(act)

    # print(onset_time)
    onsets_samples = [int(o * constants.SAMPLERATE) for o in onset_time]

    return onsets_samples


# Cut an audio files at onsets and render them
def cut_audio_files(onset_samples, audio_buffer, filename, backend: str):
    # print('audio buffer len', len(audio_buffer))
    for index, sample in enumerate(onset_samples):  # [:-1]

        # There is still an offset of x samples, TODO
        # if index == 0:
        #    sample = 0

        #print(index, ' ', sample)
        out_filename = filename.replace('.wav', '') + '_' + str(index) + backend + '.wav'
        output_file = os.path.join('cut_files', out_filename)
        logger.info('writing to ' + output_file)

        end = len(audio_buffer)
        if index != len(onset_samples) - 1:
            end = onset_samples[index + 1]
        buffer_to_write = audio_buffer[sample:end]
        # print(buffer_to_write)
        write_audio_to_path(output_file, buffer_to_write, constants.SAMPLERATE)


def trim_silence_start():
    pass


def trim_silence_end():
    pass
