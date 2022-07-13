import numpy as np

DEFAULT_SCALING_FACTOR = 0.5


def mixup(audio_features0: np.ndarray, audio_features1: np.ndarray, scaling_factor: float = DEFAULT_SCALING_FACTOR) -> \
        np.ndarray:
    """
    Mixes up two audio feature arrays, using scaling_factor as scale factor.

    :param audio_features0: a numpy array representing the first audio feature array.
    :param audio_features1: a numpy array representing the second audio feature array.
    :param scaling_factor: a float between 0 and 1 (both excluded) representing the scale factor of the mixup operation.
    :return: a numpy array obtained from the mixup operation between the first and second audio feature array,
        scaling_factor*audio_features0 + (1-scaling_factor)*audio_features1.
    :raises ValueError: if audio_features0 and audio_features1 don't share the same shape or if scaling_factor is not
        a float between 0 and 1 (both excluded).
    """
    if audio_features0.shape != audio_features1.shape:
        raise ValueError(f"Audio features arrays to mixup must share the same shape, while given shapes are "
                         f"{audio_features0.shape} and {audio_features1.shape}")
    if not 0 < scaling_factor < 1:
        raise ValueError(f"Scaling factor must be between 0 and 1, but got {scaling_factor}")
    return scaling_factor*audio_features0 + (1-scaling_factor)*audio_features1
