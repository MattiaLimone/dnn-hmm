import os
from glob import glob

import numpy
from spafe.utils import vis
import preprocessing.utils as utl
from spafe.features.lpc import lpc, lpcc

import os
from glob import glob

import numpy as np

import preprocessing.utils as utl
import pandas as pd
from tqdm.auto import tqdm
from typing import final

# init input vars
num_ceps = 13
lifter = 0
normalize = True

# read wav
#sig, fs = utl.remove_silence('data/lisa/data/timit/raw/TIMIT/TEST/DR1/FAKS0/SA1.WAV')

results = [y for x in os.walk("data/lisa/data/timit/raw/TIMIT/TRAIN") for y in glob(os.path.join(x[0], '*.WAV'))]

df_lpcc = pd.DataFrame(columns=[i for i in range(0, num_ceps * 89)])
lpccs = {"ciao": np.array([0, 0])}

for path in tqdm(results):
    filename = str(os.path.basename(path))
    data, sr = utl.remove_silence(path=path, export_path="data/cleaned/train/")
    lpccs_tosave = lpcc(sig=data, fs=sr, num_ceps=num_ceps, lifter=lifter, normalize=normalize)
    lpccs[filename] = np.array(lpccs_tosave)

np.savez("data/cleaned/train/lpccs", **lpccs)
saved = np.load("data/cleaned/train/lpccs.npz")
i = 0
for key in saved:
    if i % 100 == 0:
        print(f"{key}: " + str(saved[key]))
    i += 1


"""

# compute lpcs
lpcs = lpc(sig=sig, fs=fs, num_ceps=num_ceps)
# visualize features
vis.visualize_features(lpcs, 'LPC Index', 'Frame Index')


# visualize spectogram
vis.spectogram(sig, fs)
# compute lpccs
lpccs = lpcc(sig=sig, fs=fs, num_ceps=num_ceps, lifter=lifter, normalize=normalize)
# visualize features
vis.visualize_features(lpccs, 'LPCC Index', 'Frame Index')

print('LPC')
print(lpcs)
print('\n')
print(lpcs[0])
print('\n')
print('LPCC')
print(lpccs)
print(len(lpccs[0]))

"""

