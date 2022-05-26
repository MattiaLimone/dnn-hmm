from typing import final
import numpy as np
from hmmlearn.hmm import GMMHMM
from tqdm.auto import tqdm
from preprocessing.utils import TRAIN_PERCENTAGE

N_COMPONENTS: final = 5
N_MIX: final = 4
N_ITER: final = 12


def gmm_hmm_grid_search(X: np.ndarray, sequence_lengths: np.ndarray = None, min_state_number: int = 1,
                        max_state_number: int = 10, min_mix_number: int = 1, max_mix_number: int = 7,
                        min_iter_number: int = 10, max_iter_number: int = 11, verbose=False):
    # TODO: complete function documentation
    """
    Perform grid search to fit the best GMM-HMM model on a given speaker's audio set.
    :param X: concatenated array of the MFCCs extracted by all speaker's audio frames
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
