from spafe.features.lpc import lpcc
import numpy as np

LPCC_NUM_DEFAULT = 13
_LIFTER = 0
_NORMALIZE = True


def extract_lpcc(signal: np.ndarray, sr: int, n_lpcc: int = LPCC_NUM_DEFAULT) -> np.ndarray:
    """
        Extracts lpccs for each frame of the given audio.

        :param signal: 1 x N audio signal to extract lpccs from.
        :param sr: sample rate of audio file to construct audio frames.
        :param n_lpcc: number of lpccs to compute (default LPCC_NUM_DEFAULT)
        :return: a N x n_lpcc matrix containing lpccs for each of the given audio frames (created by sampling
                 with sampling rate sr), where N = number of constructed frames.
        """
    if sr < 0:
        raise ValueError("Sampling rate must be positive.")
    if signal.ndim != 1:
        raise ValueError("Signal must be a 1 x N mono-dimensional array.")

    lpccs = lpcc(sig=signal, fs=sr, num_ceps=n_lpcc, lifter=_LIFTER, normalize=_NORMALIZE)
    return np.array(lpccs)


def fill_audio_lpcc(audio_lpccs: np.ndarray, target_len: int, mode: int = 0) -> np.ndarray:
    """
    Fills given lpccs frame array either with 0s or repeating the coefficients circularly.

    :param audio_lpccs: lpcc frames to fill until the target size.
    :param mode: either 0 or 1, if 0 audio_lpccs will be filled with 0-valued frames, if 1 it will be filled repeating
                 audio frames in a circular way.
    :param target_len: target size of the output array.

    :return: lpcc frames filled until the target size.
    """
    if mode != 0 and mode != 1:
        raise ValueError("Mode must be either 0 or 1.")

    target_audio = np.copy(audio_lpccs)
    frame_len = len(target_audio[0])
    dist = target_len - len(target_audio)
    added_frames = 0
    fill_frame = None

    while added_frames < dist:

        if mode == 0:
            fill_frame = np.zeros(shape=(1, frame_len))
        if mode == 1:
            fill_frame = np.reshape(np.array(audio_lpccs[added_frames % len(audio_lpccs)]), newshape=(1, frame_len))

        target_audio = np.concatenate((target_audio, fill_frame), axis=0)
        added_frames += 1

    return target_audio
