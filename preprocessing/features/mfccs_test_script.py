import os
import numpy as np
import preprocessing.utils as utl
import preprocessing.features.mfcc as fe_mfcc
import pandas as pd
from tqdm.auto import tqdm
import sys
from glob import glob
from mfcc import MFCC_NUM_DEFAULT
from preprocessing.utils import fill_audio_frames

np.set_printoptions(threshold=sys.maxsize)

# Convert
results = [y for x in os.walk("data/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FCJF0") for y in
           glob(os.path.join(x[0], '*.WAV'))]
print(os.getcwd())

df_mfcc = pd.DataFrame(columns=[i for i in range(0, MFCC_NUM_DEFAULT * 3 * 89)])
mfccs = {}
mfccs_filled = {}

for path in tqdm(results):
    filename = str(os.path.basename(path))
    data, sr = utl.remove_silence(path=path)
    mfcc = fe_mfcc.extract_mfcc(signal=data, sr=sr, n_mfcc=MFCC_NUM_DEFAULT)
    mfcc = mfcc.transpose()
    mfccs[filename] = mfcc

max_len = 0
for key in tqdm(mfccs):
    if len(mfccs[key]) > max_len:
        max_len = len(mfccs[key])
print(max_len)

for key in tqdm(mfccs):
    mfccs_filled[key] = fill_audio_frames(mfccs[key], max_len, 1)

# CONTROL FINAL SHAPE
for key in tqdm(mfccs_filled):
    print(str(key) + ": " + str(len(mfccs_filled[key])))
