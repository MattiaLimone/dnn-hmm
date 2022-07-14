from typing import final
from preprocessing.constants import TRAIN_SET_PATH


# Path-related constants
MIXED_UP_MFCCS_DATASET_PATH: final = TRAIN_SET_PATH + "/mfccs_augmented.pkl"
MIXED_UP_MEL_SPEC_DATASET_PATH: final = TRAIN_SET_PATH + "/mel_spec_augmented.pkl"

# Other misc constants
MAX_MIXUPS: final = -1
