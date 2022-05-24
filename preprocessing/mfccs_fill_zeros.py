import os
import numpy as np
import preprocessing.utils as utl
import preprocessing.mfcc as fe_mfcc
import pandas as pd
from tqdm.auto import tqdm
from typing import final
import sys
from glob import glob


def fill_audio_mfcc(audio_mfccs: np.ndarray, target_len: int, mode: int = 0) -> np.ndarray:
    """
    Fills given mfccs array either with 0s or repeating the coefficients circularly.

    :param audio_mfccs: mfccs of the audio to fill
    :param mode: either 0 or 1, if 0 audio_mfccs will be filled with 0 value frames, if 1 it will be filled reapiting
                 audio frames in a circular way
    :param target_len: target size of the output array

    :return: filled audio mfccs array
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


np.set_printoptions(threshold=sys.maxsize)
MEL_COEF: final = 13

# Convert
results = [y for x in os.walk("data/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FCJF0") for y in
           glob(os.path.join(x[0], '*.WAV'))]
print(os.getcwd())

df_mfcc = pd.DataFrame(columns=[i for i in range(0, MEL_COEF * 3 * 89)])
mfccs = {}
mfccs_filled = {}

for path in tqdm(results):
    filename = str(os.path.basename(path))
    data, sr = utl.remove_silence(path=path)
    mfcc = fe_mfcc.extract_mfcc(signal=data, sr=sr, n_mfcc=MEL_COEF)
    mfcc = mfcc.transpose()
    mfccs[filename] = mfcc

max_len = 0
for key in tqdm(mfccs):
    if len(mfccs[key]) > max_len:
        max_len = len(mfccs[key])
print(max_len)

for key in tqdm(mfccs):
    mfccs_filled[key] = fill_audio_mfcc(mfccs[key], max_len, 1)

# CONTROL FINAL SHAPE
for key in tqdm(mfccs_filled):
    print(str(key) + ": " + str(len(mfccs_filled[key])))
