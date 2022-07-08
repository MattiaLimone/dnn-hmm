from typing import final

# Acoustic model-related constants
TRAIN_PERCENTAGE: final = 0.8
N_STATES_MFCCS: final = 5
N_MIX_MFCCS: final = 4
N_STATES_LPCCS: final = 4
N_MIX_LPCCS: final = 2
N_STATES_MEL_SPEC: final = 3
N_MIX_MEL_SPEC: final = 2
ACOUSTIC_MODEL_PATH: final = "acoustic_models/"
ACOUSTIC_MODEL_PATH_MFCCS: final = ACOUSTIC_MODEL_PATH + "mfccs/"
ACOUSTIC_MODEL_PATH_LPCCS: final = ACOUSTIC_MODEL_PATH + "lpccs/"
ACOUSTIC_MODEL_PATH_MEL_SPEC: final = ACOUSTIC_MODEL_PATH + "mel_spec/"

# Preprocessed train/test set-related constants
TRAIN_SET_PATH: final = "data/cleaned/train"
TEST_SET_PATH: final = "data/cleaned/test"
TRAIN_SET_PATH_MFCCS: final = TRAIN_SET_PATH + "/mfccs_train.pkl"
TEST_SET_PATH_MFCCS: final = TEST_SET_PATH + "/mfccs_test.pkl"
TRAIN_SET_PATH_LPCCS: final = TRAIN_SET_PATH + "/lpccs_train.pkl"
TEST_SET_PATH_LPCCS: final = TEST_SET_PATH + "/lpccs_test.pkl"
TRAIN_SET_PATH_MEL_SPEC: final = TRAIN_SET_PATH + "/mel_spec_train.pkl"
TEST_SET_PATH_MEL_SPEC: final = TEST_SET_PATH + "/mel_spec_test.pkl"
AUDIO_DATAFRAME_KEY: final = "Audio_Tensor"
STATE_PROB_KEY: final = "State_Probabilities"

# Original dataset-related constants
DATASET_ORIGINAL_PATH: final = "data/lisa/data/timit/raw/TIMIT/"
AUDIO_PER_SPEAKER: final = 10
