import os
from typing import final
import preprocessing.utils as utl
import preprocessing.features.mel as mel
import preprocessing.features.lpcc as lpcc
from tqdm.auto import tqdm
from preprocessing.utils import fill_audio_frames
import re


_DATASET_PATH: final = "data/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FCJF0"
_FEATURE_NAMES = {
    0: "MFCCs",
    1: "LPCCs",
    2: "Mel-scaled log-spectrogram"
}
_SPEAKER_DIR_REGEX: final = re.compile("[A-Z]{4}[0-9]")
_AUDIO_REGEX: final = re.compile("(.*)\\.WAV")


def main():
    feature_type = int(input("Insert feature type (0 for MFCCs, 1 for LPCCs, 2 for Mel-Scaled log-spectrum): "))

    try:
        feature_name = _FEATURE_NAMES[feature_type]
    except KeyError:
        raise ValueError("Feature type not understood. Must be one of the following: " + str(_FEATURE_NAMES))

    # Get audio filenames
    results = []
    for speaker_filenames in utl.speaker_audio_filenames(_DATASET_PATH, _SPEAKER_DIR_REGEX, _AUDIO_REGEX).values():
        results.extend(speaker_filenames)

    features = {}
    features_filled = {}

    for path in tqdm(results, desc=f"Extracting {feature_name}"):
        filename = str(os.path.basename(path))
        data, sr = utl.remove_silence(path=path)

        # MFCCs
        if feature_type == 0:
            mfccs_audio = mel.extract_mfccs(signal=data, sr=sr, n_mfcc=mel.MFCC_NUM_DEFAULT)
            features[filename] = mfccs_audio

        # LPCCs
        elif feature_type == 1:
            lpccs_audio = lpcc.extract_lpccs(signal=data, sr=sr, n_lpcc=lpcc.LPCC_NUM_DEFAULT)
            features[filename] = lpccs_audio

        # Mel-scaled log-spectrogram
        elif feature_type == 2:
            mel_spectrum_audio = mel.extract_mel_spectrum(signal=data, sr=sr, n_filter_bank=mel.MEL_FILTER_BANK_DEFAULT)
            features[filename] = mel_spectrum_audio

    max_len = 0
    for key in tqdm(features, desc=f"Calculating max frame length for {feature_name}"):
        if len(features[key]) > max_len:
            max_len = len(features[key])
    print(f"{feature_name} max frame length: {max_len}")

    for key in tqdm(features, desc=f"Filling {feature_name} audio frames in a circular fashion"):
        features_filled[key] = fill_audio_frames(features[key], max_len, mode=1)

    # CONTROL FINAL SHAPE
    print(f"Final shapes of {feature_name} features:")
    for key in features_filled:
        print(str(key) + ": " + str(len(features_filled[key])))


if __name__ == "__main__":
    main()
