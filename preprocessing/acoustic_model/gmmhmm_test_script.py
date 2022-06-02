import os
import sys
from glob import glob
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import preprocessing.utils as utl
from preprocessing.features.mel import extract_mfccs, MFCC_NUM_DEFAULT
from gmmhmm import generate_acoustic_model, gmm_hmm_grid_search


np.set_printoptions(threshold=sys.maxsize)
# Convert
results = [y for x in os.walk("data/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FECD0") for y in
           glob(os.path.join(x[0], '*.WAV'))]
print(os.getcwd())
print(len(results))
df_mfcc = pd.DataFrame(columns=[i for i in range(0, MFCC_NUM_DEFAULT * 3 * 89)])
mfccs = {}
train_set = []
train_set_array = np.zeros((1, 39), dtype=np.longfloat)
train_set_lengths = np.zeros(1, dtype=int)

for path in tqdm(results):
    filename = str(os.path.basename(path))
    data, sr = utl.remove_silence(path=path)
    mfcc = extract_mfccs(signal=data, sr=sr, n_mfcc=MFCC_NUM_DEFAULT)
    mfcc = mfcc.transpose()
    train_set_lengths = np.append(train_set_lengths, len(mfcc))
    train_set_array = np.concatenate((train_set_array, mfcc), axis=0)
    mfccs[filename] = mfcc

train_set_array = train_set_array[1:, ]
train_set_lengths = train_set_lengths[1:]
print("Sequence Length Shape: " + str(train_set_lengths.shape))
print("Shape: " + str(train_set_array.shape))

gmmhmm_model, all_audios_states = generate_acoustic_model(train_set_array, train_set_lengths)
print(train_set_lengths)
print([len(audio_states) for audio_states in all_audios_states])
print(gmmhmm_model.score(train_set_array, train_set_lengths))

# TODO: write automated grid search for dataset sample
best_gmmhmm_model, best_score, best_params = gmm_hmm_grid_search(train_set_array, train_set_lengths,
                                                                 min_state_number=1, max_state_number=5,
                                                                 min_mix_number=1, max_mix_number=8,
                                                                 min_iter_number=10, max_iter_number=20, verbose=False)

# best parameters seem to be 5 HMM states with 3/4 GMM mixtures each, trained in 10-12 iterations with the EM algorithm
print(best_score)
print(best_params)

# TODO: save frame-level labels corresponding to GMM-HMM most likely states extracted with viterbi algorithm
