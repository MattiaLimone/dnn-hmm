from sequentia import GMMHMM
from tqdm.auto import tqdm
import keras.models
import keras.metrics.metrics
import tensorflow as tf
from typing import final
import pandas as pd
from preprocessing.constants import TRAIN_SET_PATH_MFCCS, TEST_SET_PATH_MFCCS, AUDIO_PER_SPEAKER, AUDIO_DATAFRAME_KEY,\
    STATE_PROB_KEY, N_STATES_MFCCS
from preprocessing.file_utils import generate_or_load_speaker_ordered_dict, load_speakers_acoustic_models
from training.training_utils import TRAIN_SET_PATH_MFCCS, TEST_SET_PATH_MFCCS, TRAIN_SET_PATH_MEL_SPEC, \
    TEST_SET_PATH_MEL_SPEC, load_dataset, speaker_n_states_in_top_k_accuracy_mfccs, one_hot_labels_to_integer_labels

_EPOCHS_LOAD_RECCONV: final = 600
_VERSION_LOAD_RECCONV: final = 1.0
_RECCONV_NET_PATH: final = f"fitted_recconvsnet/recconvsnet_{_EPOCHS_LOAD_RECCONV}_epochs_v{_VERSION_LOAD_RECCONV}"

def main():

    # Load dataset
    train_mfccs, train_mfccs_labels = load_dataset(TRAIN_SET_PATH_MFCCS)
    test_mfccs, test_mfccs_labels = load_dataset(TEST_SET_PATH_MFCCS)
    train_mel_spec, train_mel_spec_labels = load_dataset(TRAIN_SET_PATH_MEL_SPEC)
    test_mel_spec, test_mel_spec_labels = load_dataset(TEST_SET_PATH_MEL_SPEC)

    labels_train = one_hot_labels_to_integer_labels(train_mfccs_labels)
    labels_test = one_hot_labels_to_integer_labels(test_mfccs_labels)

    model = keras.models.load_model(_RECCONV_NET_PATH)

    metrics = [speaker_n_states_in_top_k_accuracy_mfccs]
    model.compile(metrics=metrics)

    model.evaluate(x=[train_mfccs, train_mel_spec], y=labels_train)
    model.evaluate(x=[test_mfccs, test_mfccs_labels], y=labels_test)

if __name__ == "__main__":
    main()
