import numpy as np
from spafe.features.lpc import lpcc

NUM_CEPS = 13
LIFTER = 0
NORMALIZE = True


def export_lpcc(data, sr):

    """
    :param data: digital audio signal
    :param sr: sample rate of audio signal
    :return: linear predict cepstrum coefficients
    """

    lpccs = lpcc(sig=data, fs=sr, num_ceps=NUM_CEPS, lifter=LIFTER, normalize=NORMALIZE)
    return lpccs
