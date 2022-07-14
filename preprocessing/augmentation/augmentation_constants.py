from typing import final
from preprocessing.constants import TRAIN_SET_PATH

#
MAX_MIXUPS: final = -1

# Path-related constants
MIXED_UP_MFCCS_DATASET_PATH: final = TRAIN_SET_PATH + "/mfccs_augmented.pkl"
MIXED_UP_MEL_SPEC_DATASET_PATH: final = TRAIN_SET_PATH + "/mel_spec_augmented.pkl"
