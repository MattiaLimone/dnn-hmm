import numpy as np
from spafe.utils import preprocessing
import soundfile as sf
from typing import final
from preprocessing.constants import AUDIO_PER_SPEAKER


MINIMUM_SILENCE_LENGTH: final = 500
SILENCE_THRESHOLD: final = -35
ZERO_PADDING_MODE: final = 0
REPEATING_FRAMES_PADDING_MODE: final = 1


def remove_silence(path: str, silence_threshold: int = SILENCE_THRESHOLD) -> (np.ndarray, int):
    """
    Removes each silence frame longer than MINIMUM_SILENCE_LENGTH and not louder than the given silence threshold by an
    audio file from the given path, and then returns the silence-cleaned audio.

    :param path: a string representing the path to the audio file to remove the silence from.
    :param silence_threshold: the threshold below which audio frames are considered silence and removed.
    :return: the silence-cleaned audio file, alongside with the sample rate.
    """

    # Read the audio file
    data, sr = sf.read(path)
    energy, vad, data = preprocessing.remove_silence(sig=data, fs=sr, threshold=silence_threshold)

    return data, sr


def fill_audio_frames(audio_frames: np.ndarray, target_len: int, mode: int = ZERO_PADDING_MODE) -> np.ndarray:
    """
    Fills given audio frame array either with 0s or repeating the frames circularly.

    :param audio_frames: numpy array representing audio frame array (with each frame either containing raw sampled audio
                         data, MFCCs, LPCCs or any other kind of frame-level audio features) to fill until the target
                         size.
    :param mode: an integer, either 0 or 1, if 0 audio_frames will be filled with 0-valued frames, if 1 it will be
                 filled repeating audio frames in a circular way.
    :param target_len: an integer representing target size of the output array.

    :return: a new audio frame array filled until the target size.
    """
    if mode != ZERO_PADDING_MODE and mode != REPEATING_FRAMES_PADDING_MODE:
        raise ValueError("Mode must be either 0 or 1.")

    target_audio = np.copy(audio_frames)

    try:
        frame_len = len(target_audio[0])
    except TypeError:
        frame_len = 0  # if audio is, for example, a waveform and has shape (length, )
    dist = target_len - len(target_audio)
    added_frames = 0
    fill_frame = None

    while added_frames < dist:
        if mode == ZERO_PADDING_MODE:
            if frame_len > 0:
                fill_frame = np.zeros(shape=(1, frame_len))
            else:
                fill_frame = np.zeros(shape=(1, ))
        elif mode == REPEATING_FRAMES_PADDING_MODE:
            if frame_len > 0:
                fill_frame = np.reshape(
                    np.array(audio_frames[added_frames % len(audio_frames)]),
                    newshape=(1, frame_len)
                )
            else:
                fill_frame = np.reshape(np.array(audio_frames[added_frames % len(audio_frames)]), newshape=(1,))

        target_audio = np.concatenate((target_audio, fill_frame), axis=0)
        added_frames += 1

    return target_audio


def compute_state_frequencies(labels: np.ndarray, audios_per_speaker: int = AUDIO_PER_SPEAKER) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes state frequencies for given labels array.

    :param labels: frame-level state labels array with shape (n_samples, n_states) to compute the state frequencies for.
    :param audios_per_speaker: number of audios for each speaker.
    :return: the state ordered array, the state absolute frequencies array and the state relative frequencies array.
    """
    states, state_frequencies = np.unique(labels, return_counts=True)
    state_relative_frequencies = state_frequencies / (labels.shape[1] * audios_per_speaker)
    return states, state_frequencies, state_relative_frequencies
