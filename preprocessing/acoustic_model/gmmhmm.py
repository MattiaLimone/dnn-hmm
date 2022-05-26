import os
import sys
from glob import glob
from typing import final
import numpy as np
import pandas as pd
from hmmlearn.hmm import GMMHMM
from tqdm.auto import tqdm
import preprocessing.utils as utl
from utils import TRAIN_PERCENTAGE, MEL_COEF_NUM_DEFAULT
import mfcc

N_COMPONENTS: final = 5
N_MIX: final = 4
N_ITER: final = 12


def gmm_hmm_grid_search(X: np.ndarray, sequence_lengths: np.ndarray = None, min_state_number: int = 1,
                        max_state_number: int = 10, min_mix_number: int = 1, max_mix_number: int = 7,
                        min_iter_number: int = 10, max_iter_number: int = 11, verbose=False):
    # TODO: complete function documentation
    """
    Perform grid search to fit the best GMM-HMM model on a given speaker's audio set.
    :param X:
    :param sequence_lengths:
    :param min_state_number:
    :param max_state_number:
    :param min_mix_number:
    :param max_mix_number:
    :param min_iter_number:
    :param max_iter_number:
    :param verbose: if true, function logs info about each trained model
    :return:
    """
    n_audio_train = int(len(sequence_lengths) * TRAIN_PERCENTAGE)
    training_set_end = np.sum(sequence_lengths[:n_audio_train])
    train_set_grid_search = X[:training_set_end]
    validation_set_grid_search = X[training_set_end:]

    best_model_score = None
    best_model_params = {"n_state": None, "n_mix": None, "n_iter": None}
    best_model_grid = None
    for n_state in tqdm(range(min_state_number, max_state_number + 1)):
        for n_mix in range(min_mix_number, max_mix_number + 1):
            for n_iter in range(min_iter_number, max_iter_number + 1):
                model = GMMHMM(n_components=n_state, covariance_type='diag', n_iter=n_iter, n_mix=n_mix)
                model.fit(train_set_grid_search, sequence_lengths[:n_audio_train])
                score = model.score(validation_set_grid_search, sequence_lengths[n_audio_train:])

                if verbose:
                    print("\n\nn_mix: " + str(n_mix) + " n_state: " + str(n_state) + " n_iter: " + str(n_iter))
                    print("score: " + str(score))

                if best_model_score is None or best_model_score < score:
                    best_model_params["n_state"] = n_state
                    best_model_params["n_mix"] = n_mix
                    best_model_params["n_iter"] = n_iter
                    best_model_score = score
                    best_model_grid = model
    '''
    grid_searcher = skl.model_selection.GridSearchCV(
        estimator=GMMHMM(),
        param_grid={
            "n_components": [min_state_number, max_state_number],
            "n_mix": [min_mix_number, max_mix_number],
            "n_iter": [min_iter_number, max_iter_number]
        },
        n_jobs=max_processors,
        verbose=verbose
    )

    grid_searcher.fit(X, lengths=sequence_lengths)
    return grid_searcher.best_estimator_, grid_searcher.best_score_, grid_searcher.best_params_
    '''
    return best_model_grid, best_model_params, best_model_score


def generate_acoustic_model(X: np.ndarray, sequence_lengths: np.ndarray, n_components: int = N_COMPONENTS,
                            n_mix: int = N_MIX, n_iter=N_ITER) -> (GMMHMM, list):
    """
    Fits an acoustic GMM-HMM model on the given audio, which gives a statistical representation of the speaker's audios
    MFCCs that can be used in speaker identification context.
    :param X: concatenated array of the MFCCs extracted by all speaker's audio frames
    :param sequence_lengths: array containing the frame number of each audio in X
    :param n_components: number of HMM model states
    :param n_mix: number of GMM mixtures for each HMM state
    :param n_iter: max number of iterations of the EM algorithm used to train GMM-HMM model
    :return: trained GMM-HMM model representing the speaker's audio, a list containing the viterbi-calculated most
             likely state sequence for each audio x in X (i.e. GMM-HMM state sequence y that maximizes P(y | x))
             audio in X.
    """
    # train the GMM-HMM model on the given audios
    model = GMMHMM(n_components=n_components, covariance_type='diag', n_iter=n_iter, n_mix=n_mix)
    model.fit(X, sequence_lengths)

    audios_states = []
    sequence_start = 0
    # for each audio, apply the viterbi algorithm to get the most likely state sequence and add it to the return list
    for sequence_length in sequence_lengths:
        sequence_end = sequence_length + sequence_start
        audio = X[sequence_start:sequence_end]
        _, audio_states = model.decode(audio, algorithm='viterbi')
        audios_states.append(audio_states)
        sequence_start = sequence_end

    return model, audios_states


np.set_printoptions(threshold=sys.maxsize)
# Convert
results = [y for x in os.walk("data/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FECD0") for y in
           glob(os.path.join(x[0], '*.WAV'))]
print(os.getcwd())
print(len(results))
df_mfcc = pd.DataFrame(columns=[i for i in range(0, MEL_COEF_NUM_DEFAULT * 3 * 89)])
mfccs = {}
train_set = []
train_set_array = np.zeros((1, 39), dtype=np.longfloat)
train_set_lengths = np.zeros(1, dtype=int)

for path in tqdm(results):
    filename = str(os.path.basename(path))
    data, sr = utl.remove_silence(path=path)
    mfcc = mfcc.extract_mfcc(signal=data, sr=sr, n_mfcc=MEL_COEF_NUM_DEFAULT)
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

