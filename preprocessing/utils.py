# Loading the Libraries
import os
import numpy as np
import librosa
import librosa.display
from pydub import AudioSegment, silence
import soundfile as sf
from sklearn.decomposition import PCA
from typing import final
import matplotlib.pyplot as plt

MEL_COEF_NUM_DEFAULT: final = 13


# Function to detect and remove the silence intervals where silence
# last 500ms and decibel range reduction is higher than 16dB
def remove_silence(path: str, export_path: str = 'export/'):
    # Check if export path exist
    if not os.path.exists(export_path):
        # Create a new directory because it does not exist
        os.makedirs(export_path)
    # Read the Audiofile
    data, samplerate = librosa.load(path)
    # Name extraction from path
    filename = os.path.basename(path)
    # Save temporary file wav with rfidd
    sf.write(export_path + filename, data, samplerate)
    data_as = AudioSegment.from_wav(export_path + filename)
    # Detect silence intervals where silence last 500ms and decibel range reduction is higher than 16dB
    silence_ranges = silence.detect_silence(data_as, min_silence_len=500, silence_thresh=-16, seek_step=2)
    # Generate indexes of silence interval
    indexes = []
    for sr in silence_ranges:
        indexes = [*indexes, *range(sr[0], sr[1] + 1)]
    # Delete silence interval
    data = np.delete(data, indexes, axis=0)
    # Save wav file
    sf.write(export_path + filename, data, samplerate)
    return data, samplerate


# Function to extract Mel Frequency Cepstral Coefficient and first order and second order mfcc
def extract_mfcc(signal, sr, n_mfcc=MEL_COEF_NUM_DEFAULT, order=2):
    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, sr=sr)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    delta_mfccs = librosa.feature.delta(mfccs)
    mfccs_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))

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
    if order == 0:
        return mfccs
    if order == 1:
        return mfccs_features
    if order == 2:
        return mfccs_features



# Function to perform PCA over MFCC
def extract_mfcc_pca_features(mfcc_features):
    flattend_mfcc = np.array(mfcc_features)
    flattend_mfcc = flattend_mfcc.transpose()
    # initialize the pca
    pca = PCA(n_components=1)
    # fit the features in the model
    pca.fit(flattend_mfcc)
    # apply PCA and keep just one column, which means one feature vector with 13 features
    sample = pca.transform(flattend_mfcc)
    # reshape feature vector
    sample = sample.transpose()
    # transform it to a list in order to satisfy the format for writing the feature vector in the file
    pca_features = sample.tolist()
    # and keep just the first list, because it returns you a list of lists with only one list
    pca_features = pca_features[0]
    return pca_features


# Function to extract Linear Predictor Coefficient of n order, default = 2
def extract_lpc(signal, order=2):
    lpc_features = librosa.lpc(signal, order=order)
    return lpc_features


# Function to perform PCA over LPC
def extract_lpc_pca_features(lpc_features):
    flattend_lpc = np.array(lpc_features)
    flattend_lpc = flattend_lpc.transpose()
    # initialize the pca
    pca = PCA(n_components=1)
    # fit the features in the model
    pca.fit(flattend_lpc)
    # apply PCA and keep just one column, which means one feature vector with 13 features
    sample = pca.transform(flattend_lpc)
    # reshape feature vector
    sample = sample.transpose()
    # transform it to a list in order to satisfy the format for writing the feature vector in the file
    pca_features = sample.tolist()
    # and keep just the first list, because it returns you a list of lists with only one list
    pca_features = pca_features[0]
    return pca_features

# data, sr = remove_silence('../data/lisa/data/timit/raw/TIMIT/TEST/DR1/FAKS0/SA1.WAV')
# mfcc_feature = extract_mfcc(data, sr)
# lpc_feature = extract_lpc(data, 5)
# pca_features = extract_mfcc_pca_features(mfcc_feature)
