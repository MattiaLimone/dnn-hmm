from typing import final
import numpy as np
import torch
from lpctorch import LPCCoefficients


LPCC_NUM_DEFAULT: final = 13
_FRAME_DURATION: final = .016
_FRAME_OVERLAP: final = .5


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


    signal = torch.from_numpy(signal.reshape(-1, 1).transpose())

    # lpccs = lpcc(sig=signal, fs=sr, num_ceps=n_lpcc, win_type="hamming", lifter=_LIFTER, normalize=_NORMALIZE)

    lpc_prep = LPCCoefficients(
        sr,
        _FRAME_DURATION,
        _FRAME_OVERLAP,
        order=(n_lpcc - 1)
    )

    alphas = lpc_prep(signal)
    alphas = alphas.cpu().detach().numpy()

    lpccs = alphas[0]

    return lpccs
