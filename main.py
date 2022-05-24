import os
from glob import glob

import numpy as np

import preprocessing.utils as utl
import pandas as pd
from tqdm.auto import tqdm
from typing import final

MEL_COEF: final = 13

results = [y for x in os.walk("data/lisa/data/timit/raw/TIMIT/TRAIN") for y in glob(os.path.join(x[0], '*.WAV'))]

df_mfcc = pd.DataFrame(columns=[i for i in range(0, MEL_COEF*3*89)])
mfccs = {"ciao": np.array([0, 0])}
for path in tqdm(results):
    filename = str(os.path.basename(path))
    data, sr = utl.remove_silence(path=path, export_path="data/cleaned/train/")
    mfcc = utl.extract_mfcc(signal=data, sr=sr, n_mfcc=MEL_COEF)
    mfcc = mfcc.transpose()
    mfccs[filename] = mfcc
    print(mfcc)
    print(type(mfcc))
    #print(mfcc.shape)
    #df_mfcc.loc[-1] = mfcc_flatten

np.savez("data/cleaned/train/mfccs", **mfccs)
saved = np.load("data/cleaned/train/mfccs.npz")
i = 0
for key in saved:
    if i % 100 == 0:
        print(f"{key}: " + str(saved[key]))
    i += 1