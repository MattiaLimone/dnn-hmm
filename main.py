import os
from glob import glob
import numpy as np
import preprocessing.utils as utl
import pandas as pd
from tqdm.auto import tqdm
from typing import final
from preprocessing.utils import MEL_COEF_NUM_DEFAULT, TRAIN_PERCENTAGE

_AUDIO_PER_SPEAKER: final = 10


def speaker_label(filepath) -> str:
    """
    Extract the speaker label from the filepath, which is represented in the directory name.
    :param filepath: the speaker audio file path.
    :return: speaker label from the filepath represented in the directory name.
    """
    splitted_filepath = filepath.split('/')
    return splitted_filepath[-2]  # the dir name is the element preceding the filename in the array


results = [y for x in os.walk("data/lisa/data/timit/raw/TIMIT/") for y in glob(os.path.join(x[0], '*.WAV'))]
mfccs = {}
count_audio_speaker = 0
for path in tqdm(results):
    filename = str(os.path.basename(path))
    speaker = speaker_label(path)
    data, sr = utl.remove_silence(path=path, export_path="data/cleaned/train/")
    mfcc = utl.extract_mfcc(signal=data, sr=sr, n_mfcc=MEL_COEF_NUM_DEFAULT)
    mfcc = mfcc.transpose()
    mfccs[filename] = mfcc

    if count_audio_speaker >= _AUDIO_PER_SPEAKER:
        # if we saved all the given speaker's audio MFCCs, then reset the counter since we're working with a new speaker
        count_audio_speaker = 0

    if count_audio_speaker < int(_AUDIO_PER_SPEAKER*TRAIN_PERCENTAGE):
        # TODO: insert in the train set with corresponding label
        pass
    else:
        # TODO: insert in the test set with corresponding label
        pass
    count_audio_speaker += 1

# TODO: save both train and test set extracted MFCCs with corresponding labels
np.savez("data/cleaned/train/mfccs", **mfccs)
saved = np.load("data/cleaned/train/mfccs.npz")
i = 0
for key in saved:
    if i % 100 == 0:
        print(f"{key}: " + str(saved[key]))
    i += 1
