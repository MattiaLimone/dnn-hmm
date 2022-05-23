import os
from glob import glob
import numpy as np
import preprocessing.utils as utl
import pandas as pd
from tqdm.auto import tqdm
from typing import final
import matplotlib.pyplot as plt
import librosa
import sys
from glob import glob
from hmmlearn.hmm import GMMHMM
np.set_printoptions(threshold=sys.maxsize)
MEL_COEF: final = 13

#Convert
results = [y for x in os.walk("data/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FCJF0") for y in glob(os.path.join(x[0], '*.WAV'))]

df_mfcc = pd.DataFrame(columns=[i for i in range(0, MEL_COEF*3*89)])
mfccs = {}
mfccs_filled_zeros = {}

for path in tqdm(results):
    filename = str(os.path.basename(path))
    data, sr = utl.remove_silence(path=path)
    mfcc = utl.extract_mfcc(signal=data, sr=sr, n_mfcc=MEL_COEF)
    mfcc = mfcc.transpose()
    mfccs[filename] = mfcc

max_len = 0
for key in tqdm(mfccs):
    if len(mfccs[key]) > max_len:
        max_len = len(mfccs[key])

for key in tqdm(mfccs):
    dist = max_len - len(mfccs[key])
    if dist > 0:
        filler = np.zeros(shape=(dist, 39))
        mfccs_filled_zeros[key] = np.concatenate((mfccs[key], filler), axis=0)

# CONTROL FINAL SHAPE
for key in tqdm(mfccs_filled_zeros):
    print(str(key) + ": " + str(len(mfccs_filled_zeros[key])))