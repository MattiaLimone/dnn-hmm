import numpy as np
import librosa
from typing import final

MFCC_NUM_DEFAULT: final = 13
DERIVATIVE_ORDER_DEFAULT: final = 2


def extract_mfcc(signal: np.ndarray, sr: int, n_mfcc=MFCC_NUM_DEFAULT, order=DERIVATIVE_ORDER_DEFAULT) -> np.ndarray:
    """
    Extracts mfccs for each frame of the given audio.

    :param signal: A Numpy Array. 1 x N audio signal to extract mfccs from.
    :param sr: An integer. The sample rate of audio file to construct audio frames.
    :param n_mfcc: An integer. The number of mfccs to compute (default MFCC_NUM_DEFAULT)
    :param order: An integer. The derivative order to compute for each extracted mfcc.
    :return: a N x (n_mfcc*order) matrix containing mfccs for each of the given audio frames (created by sampling with
             sampling rate sr), where N = number of constructed frames.
    """
    if sr < 0:
        raise ValueError("Sampling rate must be positive.")
    if signal.ndim != 1:
        raise ValueError("Signal must be a 1 x N mono-dimensional array.")
    if n_mfcc <= 0:
        raise ValueError("Number of MFCCs must be strictly positive")
    if order < 0:
        raise ValueError("The derivative order must be positive")

    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, sr=sr)
    mfccs_and_deltas = np.copy(mfccs)

    for i in range(1, order + 1):
        mfccs_and_deltas = np.concatenate((mfccs_and_deltas, librosa.feature.delta(mfccs, order=order)))

    return mfccs_and_deltas.transpose()
