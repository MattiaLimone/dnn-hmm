import random
from typing import final
import tensorflow as tf
import os


# Acoustic model-related constants
N_STATES_MFCCS: final = 8  # best according to grid search
N_MIX_MFCCS: final = 3  # best according to grid search
N_STATES_LPCCS: final = 4
N_MIX_LPCCS: final = 2
N_STATES_MEL_SPEC: final = 3
N_MIX_MEL_SPEC: final = 2
ACOUSTIC_MODEL_PATH: final = "acoustic_models/"
ACOUSTIC_MODEL_PATH_MFCCS: final = ACOUSTIC_MODEL_PATH + "mfccs/"
ACOUSTIC_MODEL_PATH_LPCCS: final = ACOUSTIC_MODEL_PATH + "lpccs/"
ACOUSTIC_MODEL_PATH_MEL_SPEC: final = ACOUSTIC_MODEL_PATH + "mel_spec/"
STATE_FREQUENCIES_PATH: final = "data/state_frequencies/state_frequencies.pkl"


# Preprocessed train/test set-related constants
TRAIN_SET_PATH: final = "data/cleaned/train"
TEST_SET_PATH: final = "data/cleaned/test"
UNSPLITTED_SET_PATH: final = "data/cleaned/unsplitted"
TRAIN_SET_PATH_MFCCS: final = TRAIN_SET_PATH + "/mfccs_train.pkl"
TEST_SET_PATH_MFCCS: final = TEST_SET_PATH + "/mfccs_test.pkl"
TRAIN_SET_PATH_LPCCS: final = TRAIN_SET_PATH + "/lpccs_train.pkl"
TEST_SET_PATH_LPCCS: final = TEST_SET_PATH + "/lpccs_test.pkl"
TRAIN_SET_PATH_MEL_SPEC: final = TRAIN_SET_PATH + "/mel_spec_train.pkl"
TEST_SET_PATH_MEL_SPEC: final = TEST_SET_PATH + "/mel_spec_test.pkl"
UNSPLITTED_SET_PATH_MFCCS: final = UNSPLITTED_SET_PATH + "/mfccs.pkl"
UNSPLITTED_SET_PATH_MEL: final = UNSPLITTED_SET_PATH + "/mel_spec.pkl"

# Preprocessed tensorflow dataset-related constants
TRAIN_SET_PATH_MFCCS_TF = os.path.join(TRAIN_SET_PATH, "mfccs")
TRAIN_SET_PATH_MEL_SPEC_TF = os.path.join(TRAIN_SET_PATH, "melspec")
TRAIN_SET_PATH_LPCCS_TF = os.path.join(TRAIN_SET_PATH, "lpccs")

# Dataset keys-related constants
AUDIO_DATAFRAME_KEY: final = "Audio_Tensor"
AUDIO_NAME_DATAFRAME_KEY: final = "audioname"
SPEAKER_DATAFRAME_KEY: final = "speaker"
SAMPLE_RATE_DATAFRAME_KEY: final = "sr"
STATE_PROB_KEY: final = "State_Probabilities"


# Random-related constants
TRAIN_PERCENTAGE: final = 0.8
VALIDATION_PERCENTAGE: final = 0.2
TEST_PERCENTAGE: final = 0.2
AUTOTUNE: final = tf.data.experimental.AUTOTUNE
RANDOM_SEED: final = 47
BUFFER_SIZE: final = 1024
random.seed(RANDOM_SEED)


# Original dataset-related constants
DATASET_ORIGINAL_PATH: final = "data/lisa/data/timit/raw/TIMIT/TEST/DR1"
TRAIN_WAVEFORMS: final = os.path.join("data", "cleaned", "waveforms", "tmp")
AUDIO_PER_SPEAKER: final = 10
VOXCELEB_PATH: final = "data/voxceleb"
LONGEST_TIMIT_AUDIO_PATH = "data/dummy"
VOXCELEB_OUTPUT_PATH = "data/voxceleb"


# Audio-related constants
MAX_FRAMES_MFCCS: final = 243
MAX_FRAMES_MEL_SPEC: final = 243
MAX_FRAMES_LPCCS: final = 967
MIN_FRAMES_WAVEFORM: final = 12000
MAX_FRAMES_WAVEFORM: final = 124000

