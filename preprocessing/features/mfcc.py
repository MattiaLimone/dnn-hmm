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