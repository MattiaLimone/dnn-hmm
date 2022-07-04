import pickle
from typing import final, Union
import numpy as np
from sequentia.classifiers import GMMHMM
from tqdm.auto import tqdm
from preprocessing.constants import TRAIN_PERCENTAGE


N_COMPONENTS: final = 2
N_MIX: final = 2
N_ITER: final = 10
N_STATES: final = 5


def gmm_hmm_grid_search(X: np.ndarray, sequence_lengths: np.ndarray = None, min_state_number: int = 1,
                        max_state_number: int = 10, min_mix_number: int = 1, max_mix_number: int = 7,
                        min_iter_number: int = 10, max_iter_number: int = 11,
                        verbose: bool = False) -> (GMMHMM, list, float):
    """
    Perform grid search to fit the best GMM-HMM model on a given speaker's audio set

    :param X: A Numpy Array. The concatenated array of the MFCCs extracted by all speaker's audio frames
    :param sequence_lengths: A Numpy Array. The array of concatenated audio sequences
    :param min_state_number: An integer. The minimum number of states for grid search
    :param max_state_number: An integer. The maximum number of states for grid search
    :param min_mix_number: An integer. The minimum number of Gaussian Mixtures for grid search
    :param max_mix_number: An integer. The maximum number of Gaussian Mixtures for grid search
    :param min_iter_number: An integer. The minimum number of iterations for grid search
    :param max_iter_number: An integer. The maximum number of iterations for grid search
    :param verbose: if true, function logs info about each trained model
    :return: trained GMM-HMM model representing the speaker's audio, the params used to train the model,
            the score of the best model found.
    """
    if min_state_number > max_state_number:
        raise ValueError("Minimum state number must be less than or equal to maximum state number!")
    if min_mix_number > max_mix_number:
        raise ValueError("Minimum Gaussian mixture number must be less than or equal to "
                         "maximum Gaussian mixture number!")
    if min_iter_number > max_iter_number:
        raise ValueError("Minimum iterations number must be less than or equal to maximum iterations number!")
    if min_iter_number <= 0:
        raise ValueError("Minimum iterations number must be strictly positive.")
    if max_iter_number <= 0:
        raise ValueError("Maximum iterations number must be strictly positive.")
    if min_state_number <= 0:
        raise ValueError("Minimum states number must be strictly positive.")
    if max_state_number <= 0:
        raise ValueError("Maximum states number must be strictly positive.")
    if min_mix_number <= 0:
        raise ValueError("Minimum Gaussian mixture number must be strictly positive.")
    if max_state_number <= 0:
        raise ValueError("Maximum Gaussian mixture number must be strictly positive.")

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
                model = GMMHMM(
                    n_components=n_state,
                    covariance_type='diag',
                    n_iter=n_iter,
                    n_mix=n_mix
                )
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


def generate_acoustic_model(X: np.ndarray, label: Union[str, Union[int, float, np.number]], n_states: int = N_STATES,
                            n_mix: int = N_MIX) -> (GMMHMM, list):
    """
    Fits an acoustic GMM-HMM model on the given audio, which gives a statistical representation of the speaker's audios
    features that can be used in speaker identification context.
    :param X: A Numpy Array containing the audio features extracted by all speaker's audio frames.
    :param label: A string or numeric used as label for the model, corresponding to the class being represented.
    :param n_states: An integer. The number of HMM model states.
    :param n_mix: An integer. The number of GMM mixtures for each HMM state.
    :return: trained GMM-HMM model representing the speaker's audio, a list containing the viterbi-calculated most
             likely state sequence for each audio x in X (i.e. GMM-HMM state sequence y that maximizes P(y | x))
             audio in X.
    """
    if n_states < 0:
        raise ValueError("Components number must be positive.")
    if n_mix < 0:
        raise ValueError("The number of Gaussian mixtures  must be positive.")

    # Train the GMM-HMM model on the given audios
    model = GMMHMM(label=label, n_states=n_states, n_components=n_mix, covariance_type='diag', topology='ergodic')
    model.set_random_initial()
    model.set_random_transitions()
    model.fit(list(X))

    audios_states = []

    # For each audio, apply the viterbi algorithm to get the most likely state sequence and add it to the return list
    for i in range(0, X.shape[0]):
        audio = X[i]
        log_prob, audio_states = model.model.decode(audio, algorithm='viterbi')
        audios_states.append(audio_states)

    return model, audios_states


def save_acoustic_model(model: GMMHMM, path: str):
    """
    This function take an acoustic model and a path as input and save the given
    GMMHMM model into the given path.
    :param model: GMMHMM model to save
    :param path: path where GMMHMM model will be saved
    """
    with open(path, "wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_acoustic_model(path: str) -> GMMHMM:
    """
    This function takes in input a path and return the GMMHMM
     acoustic model saved from this path
    :param path: path of the acoustic model
    :return: GMMHMM acoustic model
    """
    with open(path, "rb") as file:
        acoustic_model = pickle.load(file)
        return acoustic_model
