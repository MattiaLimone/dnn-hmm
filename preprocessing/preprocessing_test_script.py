import os
import re
from typing import final, Optional
from tqdm.auto import tqdm
import utils as utl
from features.mfcc import extract_mfcc, MFCC_NUM_DEFAULT
from features.lpcc import extract_lpcc, LPCC_NUM_DEFAULT
from acoustic_model.gmmhmm import generate_acoustic_model
import numpy as np
import pandas as pd


_DATASET_PATH: final = "data/lisa/data/timit/raw/TIMIT"
_SPEAKER_DIR_REGEX: final = re.compile("[A-Z]{4}[0-9]")
_AUDIO_REGEX: final = re.compile("(.*)\\.WAV")
_AUDIO_PER_SPEAKER: final = 10


def _speakers_audios_filename(path: str, speakers_audios: dict, visited: Optional[set] = None):
    if visited is None:
        visited = set()
    basename = os.path.basename(path)

    if os.path.isdir(path) and basename not in visited:
        visited.add(path)

        # Base case: leaf in searched files
        if _SPEAKER_DIR_REGEX.match(basename):

            if basename not in speakers_audios:
                speakers_audios[basename] = []

            for entry in os.listdir(path):
                if _AUDIO_REGEX.match(entry):
                    audio_path = path + entry
                    speakers_audios[basename].append(audio_path)

        # Recursive call
        else:
            for entry in os.listdir(path):

                newpath = path + "/" + entry
                if os.path.isdir(newpath):
                    _speakers_audios_filename(newpath, speakers_audios, visited)


def _speakers_audios_mfccs_lpccs_max_frames(speakers_audios_names: dict) -> (dict, dict, int):
    speaker_audios_mfccs = {}
    speaker_audios_lpccs = {}
    max_frames = 0

    # For each speaker, extract MFCCs and LPCCs and search for max frame number
    for speaker in tqdm(speakers_audios_names):

        if speaker not in speaker_audios_mfccs:
            speaker_audios_mfccs[speaker] = []

        if speaker not in speaker_audios_mfccs:
            speaker_audios_lpccs[speaker] = []

        for audio_path in speakers_audios_names[speaker]:
            silence_cleaned_audio, sr = utl.remove_silence(audio_path)

            # MFCCs handling
            mfccs = extract_mfcc(silence_cleaned_audio, sr)
            speaker_audios_mfccs[speaker].append(mfccs)

            # LPCCs handling
            lpccs = extract_lpcc(silence_cleaned_audio, sr)
            speaker_audios_lpccs[speaker].append(lpccs)

            # Update max frame num if found higher number of frames
            if len(mfccs) > max_frames:
                max_frames = len(mfccs)

            return speaker_audios_mfccs, speaker_audios_lpccs, max_frames


def _fill_all_speaker_audios(speaker_audios_mfccs, speaker_audios_lpccs, max_frames) -> (dict, dict, dict, dict):
    speaker_audios_lpcc_filled_zeros = {}
    speaker_audios_mfcc_filled_zeros = {}
    speaker_audios_lpcc_filled_circular = {}
    speaker_audios_mfcc_filled_circular = {}

    for speaker in speaker_audios_mfccs:
        # Create numpy _AUDIO_PER_SPEAKER x max_frames x MFCC_NUM_DEFAULT tensor to contain
        speaker_audios_mfcc_filled_zeros[speaker] = np.zeros(
            shape=(max_frames, MFCC_NUM_DEFAULT, 1),
            dtype=np.float64
        )

        for speaker_audio in speaker_audios_mfccs[speaker]:
            # Fill with zeros
            filled_audio = utl.fill_audio_frames(speaker_audio, target_len=max_frames, mode=0)
            speaker_audios_mfcc_filled_zeros[speaker] = np.append(
                speaker_audios_mfcc_filled_zeros[speaker],
                filled_audio.reshape(shape=(1, max_frames, MFCC_NUM_DEFAULT)),
                axis=0
            )

            # Fill in a circular fashion
            filled_audio = utl.fill_audio_frames(speaker_audio, target_len=max_frames, mode=1)
            speaker_audios_mfcc_filled_circular[speaker] = np.append(
                speaker_audios_mfcc_filled_circular[speaker],
                filled_audio.reshape(shape=(1, max_frames, MFCC_NUM_DEFAULT)),
                axis=0
            )

        for speaker_audio in speaker_audios_lpccs[speaker]:
            # Fill with zeros
            filled_audio = utl.fill_audio_frames(speaker_audio, target_len=max_frames, mode=0)
            speaker_audios_lpcc_filled_zeros[speaker] = np.append(
                speaker_audios_lpcc_filled_zeros[speaker],
                filled_audio.reshape(shape=(1, max_frames, LPCC_NUM_DEFAULT)),
                axis=0
            )

            # Fill in a circular fashion
            filled_audio = utl.fill_audio_frames(speaker_audio, target_len=max_frames, mode=1)
            speaker_audios_lpcc_filled_circular[speaker] = np.append(
                speaker_audios_lpcc_filled_circular[speaker],
                filled_audio.reshape(shape=(1, max_frames, LPCC_NUM_DEFAULT)),
                axis=0
            )

    return speaker_audios_mfcc_filled_zeros, speaker_audios_lpcc_filled_zeros, \
           speaker_audios_mfcc_filled_circular, speaker_audios_lpcc_filled_circular


def _generate_speakers_acoustic_model(speakers_audios_features: dict) -> (dict, dict):
    acoustic_models = {}
    acoustic_model_state_labels = {}

    for speaker in speakers_audios_features:
        # Flatten feature matrix into array of frame features
        speaker_audios = speakers_audios_features[speaker]
        audio_length = speaker_audios.shape[1]
        feature_number = speaker_audios.shape[2]
        number_of_audios = len(speaker_audios)
        speaker_audios = speaker_audios.reshape(shape=(number_of_audios * audio_length, feature_number))

        # Extract acoustic models and frame-level labels (most likely sequence of states from viterbi algorithm)
        audio_lengths = np.array([audio_length for i in range(0, number_of_audios)])
        acoustic_models[speaker], acoustic_model_state_labels[speaker] = generate_acoustic_model(
            speaker_audios, audio_lengths
        )
    return acoustic_models, acoustic_model_state_labels


def _one_hot_encode_state_labels(speakers_state_labels: dict):
    return 0


def main():
    speakers_audios_names = {}

    # Get audio paths, grouped by speaker
    _speakers_audios_filename(_DATASET_PATH, speakers_audios_names)

    # Get max frame len, audio MFCCs and LPCCs for each speaker
    speaker_audios_mfccs, speaker_audios_lpccs, max_frames = _speakers_audios_mfccs_lpccs_max_frames(
        speakers_audios_names
    )

    # Normalize length of all audio sequences
    speaker_audios_mfcc_filled_zeros, speaker_audios_lpcc_filled_zeros, speaker_audios_lpcc_filled_circular, \
    speaker_audios_mfcc_filled_circular = _fill_all_speaker_audios(
        speaker_audios_mfccs,
        speaker_audios_lpccs,
        max_frames
    )

    # Construct acoustic models and extract frame-level labels for each variation of the features (mfccs, lpccs)
    acoustic_models_mfcc_filled_zeros, labels_mfcc_filled_zeros = _generate_speakers_acoustic_model(
        speaker_audios_mfcc_filled_zeros
    )
    acoustic_models_lpcc_filled_zeros, labels_lpcc_filled_zeros = _generate_speakers_acoustic_model(
        speaker_audios_lpcc_filled_zeros
    )
    acoustic_models_mfcc_filled_circular, labels_mfcc_filled_circular = _generate_speakers_acoustic_model(
        speaker_audios_mfcc_filled_circular
    )
    acoustic_models_lpcc_filled_circular, labels_lpcc_filled_circular = _generate_speakers_acoustic_model(
        speaker_audios_lpcc_filled_circular
    )

    # TODO: one-hot encode frame-level state labels, taking into account that each label must be
    #  (n_states*speaker_index) + state_index

    # TODO: save extracted features and labels to npz files in a suitable format for model training
    df = pd.DataFrame(columns=["Audio", "States"])


if __name__ == "__main__":
    main()
