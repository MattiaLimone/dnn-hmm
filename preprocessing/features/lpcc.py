from spafe.features.lpc import lpcc
import numpy as np

LPCC_NUM_DEFAULT = 13
_LIFTER = 0
_NORMALIZE = True


def extract_lpccs(signal: np.ndarray, sr: int, n_lpcc: int = LPCC_NUM_DEFAULT) -> np.ndarray:
    """
        Extracts LPCCs for each frame of the given audio.

        :param signal: A Numpy Array. 1 x N audio signal to extract LPCCs from.
        :param sr: An integer. The sample rate of audio file to construct audio frames.
        :param n_lpcc: An integer. The number of LPCCs to compute (default LPCC_NUM_DEFAULT)
        :return: a N x n_lpcc matrix containing LPCCs for each of the given audio frames (created by sampling
                 with sampling rate sr), where N = number of constructed frames.
        """
    if sr < 0:
        raise ValueError("Sampling rate must be positive.")
    if signal.ndim != 1:
        raise ValueError("Signal must be a 1 x N mono-dimensional array.")
    if n_lpcc <= 0:
        raise ValueError("Number of LPCCs must be strictly positive")

    lpccs = lpcc(sig=signal, fs=sr, num_ceps=n_lpcc, lifter=_LIFTER, normalize=_NORMALIZE)
    return np.array(lpccs)
