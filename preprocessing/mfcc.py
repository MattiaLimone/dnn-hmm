import numpy as np
import librosa
from typing import final

MEL_COEF_NUM_DEFAULT: final = 13

# Function to extract Mel Frequency Cepstral Coefficient and first order and second order mfcc
def extract_mfcc(signal: np.ndarray, sr: int, n_mfcc=MEL_COEF_NUM_DEFAULT, order=2):
    """
    :param signal: digital signal of audio file
    :param sr: sample rate of audio file
    :param n_mfcc: number of mel coefficient to compute (default 13)
    :param order: derivate order
    :return: mel frequency cepstrum coefficients
    """

    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, sr=sr)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    delta_mfccs = librosa.feature.delta(mfccs)
    mfccs_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))

    """
    #Plot Mfccs
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(mfccs,
                             x_axis="time",
                             sr=sr)
    plt.colorbar(format="%+2.f")
    plt.show()
    # Plot first derivative Mfccs
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(delta2_mfccs,
                             x_axis="time",
                             sr=sr)
    plt.colorbar(format="%+2.f")
    plt.show()
    # Plot second derivative Mfccs
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(delta_mfccs,
                             x_axis="time",
                             sr=sr)
    plt.colorbar(format="%+2.f")
    plt.show()
    """

    if order == 0:
        return mfccs
    if order == 1:
        return mfccs_features
    if order == 2:
        return mfccs_features
