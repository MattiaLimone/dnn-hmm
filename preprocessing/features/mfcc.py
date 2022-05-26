import numpy as np
import librosa
from typing import final


MFCC_NUM_DEFAULT: final = 13
DERIVATIVE_ORDER_DEFAULT: final = 2


# Function to extract Mel Frequency Cepstral Coefficient and first order and second order mfcc
def extract_mfcc(signal: np.ndarray, sr: int, n_mfcc=MFCC_NUM_DEFAULT, order=DERIVATIVE_ORDER_DEFAULT):
    """
    Extracts mfccs for each frame of the given audio.

    :param signal: 1 x N audio signal to extract mfccs from.
    :param sr: sample rate of audio file to construct audio frames.
    :param n_mfcc: number of mfccs to compute (default MFCC_NUM_DEFAULT)
    :param order: derivative order to compute for each extracted mfcc.
    :return: a N x (n_mfcc*order) matrix containing mfccs for each of the given audio frames (created by sampling with
             sampling rate sr), where N = number of constructed frames.
    """

    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, sr=sr)
    mfccs_and_deltas = np.copy(mfccs)

    for i in range(1, order + 1):
        mfccs_and_deltas = np.concatenate((mfccs_and_deltas, librosa.feature.delta(mfccs, order=order)))

    return mfccs_and_deltas


def fill_audio_mfcc(audio_mfccs: np.ndarray, target_len: int, mode: int = 0) -> np.ndarray:
    """
    Fills given mfccs frame array either with 0s or repeating the coefficients circularly.

    :param audio_mfccs: mfcc frames to fill until the target size.
    :param mode: either 0 or 1, if 0 audio_mfccs will be filled with 0-valued frames, if 1 it will be filled repeating
                 audio frames in a circular way.
    :param target_len: target size of the output array.

    :return: mfcc frames filled until the target size.
    """
    if mode != 0 and mode != 1:
        raise ValueError("Mode must be either 0 or 1.")

    target_audio = np.copy(audio_mfccs)
    frame_len = len(target_audio[0])
    dist = target_len - len(target_audio)
    added_frames = 0
    fill_frame = None

    while added_frames < dist:

        if mode == 0:
            fill_frame = np.zeros(shape=(1, frame_len))
        if mode == 1:
            fill_frame = np.reshape(np.array(audio_mfccs[added_frames % len(audio_mfccs)]), newshape=(1, frame_len))

        target_audio = np.concatenate((target_audio, fill_frame), axis=0)
        added_frames += 1

    return target_audio


