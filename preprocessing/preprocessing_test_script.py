import os
import re
from typing import final, Optional
from tqdm.auto import tqdm
import utils as utl
from features.mfcc import extract_mfcc, MFCC_NUM_DEFAULT
from features.lpcc import extract_lpcc, LPCC_NUM_DEFAULT
from acoustic_model.gmmhmm import generate_acoustic_model, N_COMPONENTS as N_STATES
import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict
import sklearn as skl

_DATASET_PATH: final = "data/lisa/data/timit/raw/TIMIT"
_SPEAKER_INDEXES_PATH: final = "data/lisa/data/speakerindexes.pkl"
_TRAIN_SET_PATH: final = "data/cleaned/train"
_TEST_SET_PATH: final = "data/cleaned/test"
_SPEAKER_DIR_REGEX: final = re.compile("[A-Z]{4}[0-9]")
_AUDIO_REGEX: final = re.compile("(.*)\\.WAV")
_AUDIO_PER_SPEAKER: final = 10
_AUDIO_DATAFRAME_KEY = "Audio_Tensor"
_STATE_PROB_KEY = "State_Probabilities"
_RANDOM_SEED = 47


def _generate_or_load_speaker_ordered_dict(speakers: list, generate: bool = False) -> dict:
    speaker_indexes = OrderedDict()
    generate_flag = generate

    # If generate flag is False, try to load speaker indexes file, otherwise set generate flag to True
    if not generate:
        try:
            with open(_SPEAKER_INDEXES_PATH, "rb") as file:
                speaker_indexes = pickle.load(file)
        except IOError:
            generate_flag = True

    # If generate flag is True, generate new speaker indexes OrderedDict and save it to file with pickle
    if generate_flag:
        for i in range(0, len(speakers)):
            speaker_indexes[speakers[i]] = i

        with open(_SPEAKER_INDEXES_PATH, "wb") as file:
            pickle.dump(speaker_indexes, file, protocol=pickle.HIGHEST_PROTOCOL)

    return speaker_indexes


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


def _fill_speakers_audios_features(speaker_audio_features: dict, max_frames: int, feature_num: int,
                                   mode: int = 0) -> dict:
    speaker_audios_features_filled = {}

    for speaker in speaker_audio_features:
        # Create empty numpy _AUDIO_PER_SPEAKER x max_frames x feature_num tensor to contain the filled audio features
        speaker_audios_features_filled[speaker] = np.zeros(
            shape=(1, max_frames, feature_num),
            dtype=np.float64
        )

        # For each audio of the speaker
        for speaker_audio in speaker_audios_features_filled[speaker]:
            # Fill with zero-value features or in a circular fashion based on the given mode
            filled_audio = utl.fill_audio_frames(speaker_audio, target_len=max_frames, mode=mode)
            speaker_audios_features_filled[speaker] = np.append(
                speaker_audios_features_filled[speaker],
                filled_audio.reshape(shape=(1, max_frames, feature_num)),
                axis=0
            )

        # Remove first, empty matrix from the result tensor
        speaker_audios_features_filled[speaker] = speaker_audios_features_filled[speaker][1:]

        return speaker_audios_features_filled


def _fill_all_speaker_audios(speaker_audios_mfccs: dict, speaker_audios_lpccs: dict, max_frames: int) -> (
dict, dict, dict, dict):
    speaker_audios_lpcc_filled_zeros = {}
    speaker_audios_mfcc_filled_zeros = {}
    speaker_audios_lpcc_filled_circular = {}
    speaker_audios_mfcc_filled_circular = {}

    # For each speaker
    for speaker in speaker_audios_mfccs:
        # Create empty numpy _AUDIO_PER_SPEAKER x max_frames x MFCC_NUM_DEFAULT tensor to contain the filled audio MFCCs
        speaker_audios_mfcc_filled_zeros[speaker] = np.zeros(
            shape=(1, max_frames, MFCC_NUM_DEFAULT),
            dtype=np.float64
        )

        speaker_audios_mfcc_filled_circular[speaker] = np.zeros(
            shape=(1, max_frames, MFCC_NUM_DEFAULT),
            dtype=np.float64
        )

        # For each audio of the speaker
        for speaker_audio in speaker_audios_mfccs[speaker]:
            # Fill with zero-value MFCCs
            filled_audio = utl.fill_audio_frames(speaker_audio, target_len=max_frames, mode=0)
            speaker_audios_mfcc_filled_zeros[speaker] = np.append(
                speaker_audios_mfcc_filled_zeros[speaker],
                filled_audio.reshape(shape=(1, max_frames, MFCC_NUM_DEFAULT)),
                axis=0
            )

            # Fill in a circular fashion repeating MFCCs
            filled_audio = utl.fill_audio_frames(speaker_audio, target_len=max_frames, mode=1)
            speaker_audios_mfcc_filled_circular[speaker] = np.append(
                speaker_audios_mfcc_filled_circular[speaker],
                filled_audio.reshape(shape=(1, max_frames, MFCC_NUM_DEFAULT)),
                axis=0
            )

        # Remove first, empty matrices from the result tensor
        speaker_audios_mfcc_filled_zeros[speaker] = speaker_audios_mfcc_filled_circular[speaker][1:]
        speaker_audios_mfcc_filled_circular[speaker] = speaker_audios_mfcc_filled_zeros[speaker][1:]

        # Create empty numpy _AUDIO_PER_SPEAKER x max_frames x LPCC_NUM_DEFAULT tensor to contain the filled audio LPCCs
        speaker_audios_lpcc_filled_zeros[speaker] = np.zeros(
            shape=(1, max_frames, LPCC_NUM_DEFAULT),
            dtype=np.float64
        )

        speaker_audios_lpcc_filled_circular[speaker] = np.zeros(
            shape=(1, max_frames, LPCC_NUM_DEFAULT),
            dtype=np.float64
        )

        # For each audio of the speaker
        for speaker_audio in speaker_audios_lpccs[speaker]:
            # Fill with zero-valued LPCCs
            filled_audio = utl.fill_audio_frames(speaker_audio, target_len=max_frames, mode=0)
            speaker_audios_lpcc_filled_zeros[speaker] = np.append(
                speaker_audios_lpcc_filled_zeros[speaker],
                filled_audio.reshape(shape=(1, max_frames, LPCC_NUM_DEFAULT)),
                axis=0
            )

            # Fill in a circular fashion repeating LPCCs
            filled_audio = utl.fill_audio_frames(speaker_audio, target_len=max_frames, mode=1)
            speaker_audios_lpcc_filled_circular[speaker] = np.append(
                speaker_audios_lpcc_filled_circular[speaker],
                filled_audio.reshape(shape=(1, max_frames, LPCC_NUM_DEFAULT)),
                axis=0
            )

        # Remove first, empty matrices from the result tensor
        speaker_audios_lpcc_filled_zeros[speaker] = speaker_audios_lpcc_filled_circular[speaker][1:]
        speaker_audios_lpcc_filled_circular[speaker] = speaker_audios_lpcc_filled_zeros[speaker][1:]

    return speaker_audios_mfcc_filled_zeros, speaker_audios_lpcc_filled_zeros, \
           speaker_audios_mfcc_filled_circular, speaker_audios_lpcc_filled_circular


def _generate_speakers_acoustic_model(speakers_audios_features: dict) -> (dict, dict):
    acoustic_models = {}
    acoustic_model_state_labels = {}

    # For each speaker
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


def _one_hot_encode_state_labels(speakers_raw_state_labels: dict, speaker_indexes: dict) -> np.ndarray:
    speakers_global_state_labels = {}
    n_audio = 0  # audio number counter
    max_frames = 0  # maximum number of frames

    # For each speaker
    for speaker in speaker_indexes:
        speakers_global_state_labels[speaker] = []

        # For each audio state label array of the speaker
        for raw_audio_state_labels in speakers_raw_state_labels[speaker]:
            global_audio_state_labels = np.array([])

            # Increment audio number counter
            n_audio += 1

            # Update frame max length if needed
            if max_frames < len(raw_audio_state_labels):
                max_frames = len(raw_audio_state_labels)

            # For all state labels of an audio, replace raw state index with global state index with formula:
            # (n_states*speaker_index) + raw_state_label
            for raw_state_label in raw_audio_state_labels:
                global_state_label = (N_STATES * speaker_indexes[speaker]) + raw_state_label
                global_audio_state_labels = np.append(global_audio_state_labels, global_state_label)

            speakers_global_state_labels[speaker].append(global_audio_state_labels)

    n_speaker = len(speakers_global_state_labels)

    # Create n_audio x max_frames x (n_states*n_speaker) tensor representing the one-hot encoding of the state labels
    one_hot_encoded_state_labels = np.zeros(shape=(n_audio, max_frames, N_STATES * n_speaker))

    audio_index = 0  # audio counter to index the one-encoding 3rd-order tensor

    # For each speaker
    for speaker in speaker_indexes:

        # For each audio of the speaker
        for global_audio_state_labels in speakers_global_state_labels[speaker]:

            # For each frame of the audio
            for frame_index in range(0, len(global_audio_state_labels)):
                # Get the target most likely state for the frame according to the viterbi algorithm
                state_index = global_audio_state_labels[frame_index]

                # Set the corresponding component of the one-hot encode label vector to 1
                one_hot_encoded_state_labels[audio_index, frame_index, state_index] = 1
            audio_index += 1

    return one_hot_encoded_state_labels


def _generate_audios_feature_tensor(speaker_audios_features: dict, speaker_indexes: dict) -> np.ndarray:
    # Create n_audio x max_frames x (n_features) tensor to contain the feature matrix of each audio
    audios_feature_tensor = None

    # For each speaker
    for speaker in speaker_indexes:
        # Get the feature matrix for each audio of the speaker, and concatenate it to the final feature tensor
        speaker_audios_feature_tensor = speaker_audios_features[speaker]

        # If output tensor is still empty, then copy the feature speaker's audios feature tensor into output tensor,
        # otherwise concatenate it to the current output tensor
        if audios_feature_tensor is None:
            audios_feature_tensor = np.copy(speaker_audios_feature_tensor)
        else:
            audios_feature_tensor = np.concatenate((audios_feature_tensor, speaker_audios_feature_tensor), axis=0)

    return audios_feature_tensor


def _generate_output_dataframe(audios_feature_tensor: np.ndarray, one_hot_encoded_labels: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(columns=[_AUDIO_DATAFRAME_KEY, _STATE_PROB_KEY])
    df[_AUDIO_DATAFRAME_KEY] = df[_AUDIO_DATAFRAME_KEY].astype(np.ndarray)
    df[_STATE_PROB_KEY] = df[_STATE_PROB_KEY].astype(np.ndarray)

    # For each audio feature matrix
    for i in range(0, len(audios_feature_tensor)):
        # Get feature matrix and state labels probabilities, putting them into a pandas dataframe
        df.loc[i] = (audios_feature_tensor[i], one_hot_encoded_labels[i])

    return df


def main():
    speakers_audios_names = {}

    # Get audio paths, grouped by speaker
    _speakers_audios_filename(_DATASET_PATH, speakers_audios_names)

    # Generate the speaker indexes to grant the speakers are always processed in the same order (or load it if saved)
    speaker_indexes = _generate_or_load_speaker_ordered_dict(list(speakers_audios_names.keys()))

    # Get max frame len, audio MFCCs and LPCCs for each speaker
    speaker_audios_mfccs, speaker_audios_lpccs, max_frames = _speakers_audios_mfccs_lpccs_max_frames(
        speakers_audios_names
    )

    # Normalize length for MFCC
    speaker_audios_mfcc_filled_zeros = _fill_speakers_audios_features(speaker_audios_mfccs, max_frames,
                                                                      MFCC_NUM_DEFAULT, 0)
    speaker_audios_mfcc_filled_circular = _fill_speakers_audios_features(speaker_audios_mfccs, max_frames,
                                                                         LPCC_NUM_DEFAULT, 1)

    # Normalize length for LPCC
    speaker_audios_lpcc_filled_zeros = _fill_speakers_audios_features(speaker_audios_lpccs, max_frames,
                                                                      MFCC_NUM_DEFAULT, 0)
    speaker_audios_lpcc_filled_circular = _fill_speakers_audios_features(speaker_audios_lpccs, max_frames,
                                                                         LPCC_NUM_DEFAULT, 1)

    """
     # Normalize length of all audio sequences
    speaker_audios_mfcc_filled_zeros, speaker_audios_lpcc_filled_zeros, speaker_audios_lpcc_filled_circular, \
    speaker_audios_mfcc_filled_circular = _fill_all_speaker_audios(
        speaker_audios_mfccs,
        speaker_audios_lpccs,
        max_frames
    )
    """

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

    # One-hot encode frame-level state labels as vectors
    one_hot_encoded_labels_mfcc_filled_zeros = _one_hot_encode_state_labels(
        labels_mfcc_filled_zeros,
        speaker_indexes
    )
    one_hot_encoded_labels_lpcc_filled_zeros = _one_hot_encode_state_labels(
        labels_lpcc_filled_zeros,
        speaker_indexes
    )
    one_hot_encoded_labels_mfcc_filled_circular = _one_hot_encode_state_labels(
        labels_mfcc_filled_circular,
        speaker_indexes
    )
    one_hot_encoded_labels_lpcc_filled_circular = _one_hot_encode_state_labels(
        labels_lpcc_filled_circular,
        speaker_indexes
    )

    # Construct the audio feature tensor for both MFCCs and LPCCs
    audios_feature_tensor_mfcc_filled_zeros = _generate_audios_feature_tensor(
        speaker_audios_mfcc_filled_zeros,
        speaker_indexes
    )
    audios_feature_tensor_lpcc_filled_zeros = _generate_audios_feature_tensor(
        speaker_audios_lpcc_filled_zeros,
        speaker_indexes
    )
    audios_feature_tensor_mfcc_filled_circular = _generate_audios_feature_tensor(
        speaker_audios_mfcc_filled_circular,
        speaker_indexes
    )
    audios_feature_tensor_lpcc_filled_circular = _generate_audios_feature_tensor(
        speaker_audios_lpcc_filled_circular,
        speaker_indexes
    )

    # Generate dataframes containing input (feature) and output (label) tensors
    df_mfcc_filled_zeros = _generate_output_dataframe(
        audios_feature_tensor_mfcc_filled_zeros,
        one_hot_encoded_labels_mfcc_filled_zeros
    )
    df_lpcc_filled_zeros = _generate_output_dataframe(
        audios_feature_tensor_lpcc_filled_zeros,
        one_hot_encoded_labels_lpcc_filled_zeros
    )
    df_mfcc_filled_circular = _generate_output_dataframe(
        audios_feature_tensor_mfcc_filled_circular,
        one_hot_encoded_labels_mfcc_filled_circular
    )
    df_lpcc_filled_circular = _generate_output_dataframe(
        audios_feature_tensor_lpcc_filled_circular,
        one_hot_encoded_labels_lpcc_filled_circular
    )

    # Split generated dataframes into train and test sets
    df_mfcc_filled_zeros_train, df_mfcc_filled_zeros_test = skl.model_selection.train_test_split(
        df_mfcc_filled_zeros,
        train_size=utl.TRAIN_PERCENTAGE,
        shuffle=True,
        random_state=_RANDOM_SEED
    )
    df_lpcc_filled_zeros_train, df_lpcc_filled_zeros_test = skl.model_selection.train_test_split(
        df_lpcc_filled_zeros,
        train_size=utl.TRAIN_PERCENTAGE,
        shuffle=True,
        random_state=_RANDOM_SEED
    )
    df_mfcc_filled_circular_train, df_mfcc_filled_circular_test = skl.model_selection.train_test_split(
        df_mfcc_filled_circular,
        train_size=utl.TRAIN_PERCENTAGE,
        shuffle=True,
        random_state=_RANDOM_SEED
    )
    df_lpcc_filled_circular_train, df_lpcc_filled_circular_test = skl.model_selection.train_test_split(
        df_lpcc_filled_circular,
        train_size=utl.TRAIN_PERCENTAGE,
        shuffle=True,
        random_state=_RANDOM_SEED
    )

    # Save extracted features and labels to pickle files in a suitable format for model training
    df_mfcc_filled_zeros_train.to_pickle(_TRAIN_SET_PATH + "/mfccs_filled_zeros_train.pkl")
    df_lpcc_filled_zeros_train.to_pickle(_TRAIN_SET_PATH + "/lpccs_filled_zeros_train.pkl")
    df_mfcc_filled_circular_train.to_pickle(_TRAIN_SET_PATH + "/mfccs_filled_circular_train.pkl")
    df_lpcc_filled_circular_train.to_pickle(_TRAIN_SET_PATH + "/lpccs_filled_circular_train.pkl")

    df_mfcc_filled_zeros_test.to_pickle(_TRAIN_SET_PATH + "/mfccs_filled_zeros_test.pkl")
    df_lpcc_filled_zeros_test.to_pickle(_TRAIN_SET_PATH + "/lpccs_filled_zeros_test.pkl")
    df_mfcc_filled_circular_test.to_pickle(_TRAIN_SET_PATH + "/mfccs_filled_circular_test.pkl")
    df_lpcc_filled_circular_test.to_pickle(_TRAIN_SET_PATH + "/lpccs_filled_circular_test.pkl")


if __name__ == "__main__":
    main()
