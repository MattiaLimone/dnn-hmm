from typing import final
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import sklearn as skl
import scipy.sparse as sp
import utils as utl
from features.mel import extract_mfccs, extract_mel_spectrum, MFCC_NUM_DEFAULT, MEL_FILTER_BANK_DEFAULT, \
    DERIVATIVE_ORDER_DEFAULT
from features.lpcc import extract_lpccs, LPCC_NUM_DEFAULT
from acoustic_model.gmmhmm import generate_acoustic_model, save_acoustic_model
from preprocessing.constants import ACOUSTIC_MODEL_PATH, TRAIN_PERCENTAGE, AUDIO_DATAFRAME_KEY, \
    STATE_PROB_KEY, N_STATES_MFCCS, N_MIX_MFCCS, N_STATES_LPCCS, N_MIX_LPCCS, N_STATES_MEL_SPEC, N_MIX_MEL_SPEC, \
    DATASET_ORIGINAL_PATH, TRAIN_SET_PATH_MFCCS, TRAIN_SET_PATH_LPCCS, \
    TRAIN_SET_PATH_MEL_SPEC, TEST_SET_PATH_MFCCS, TEST_SET_PATH_LPCCS, TEST_SET_PATH_MEL_SPEC
from preprocessing.file_utils import speaker_audio_filenames, generate_or_load_speaker_ordered_dict, \
    SPEAKER_DIR_REGEX, AUDIO_REGEX


_RANDOM_SEED: final = 47


def _speakers_audios_mfccs_lpccs_mel_spectrograms_max_frames(speakers_audios_names: dict) -> (dict, dict, dict, int,
                                                                                              int, int):
    """
    Computes the maximum frames length among all audios, then it generates a dictionary containing speaker-MFCCs pairs
    and a dictionary containing speaker-LPCCs, speaker-Mel-scaled-log-spectrogram pairs (for each audio).

    :param speakers_audios_names: A dictionary of speaker-path_to_audio_files pairs.
    :return: A dictionary containing speaker-MFCCs pairs, a dictionary containing speaker-LPCCs pairs, a dictionary
     containing speaker-Mel-scaled-log-spectrogram pairs (for each audio) and the maximum frames length.
    """
    speaker_audios_mfccs = {}
    speaker_audios_lpccs = {}
    speaker_audios_mel_spectrogram = {}
    max_frames_mfcc = 0
    max_frames_lpcc = 0
    max_frames_mel = 0

    # For each speaker
    for speaker in tqdm(speakers_audios_names, desc="Extracting MFCCs, LPCCs, Mel Spectrogram, Max Frame: "):

        if speaker not in speaker_audios_mfccs:
            speaker_audios_mfccs[speaker] = []

        if speaker not in speaker_audios_lpccs:
            speaker_audios_lpccs[speaker] = []

        if speaker not in speaker_audios_mel_spectrogram:
            speaker_audios_mel_spectrogram[speaker] = []

        # Extract MFCCs and LPCCs and search for max frame number
        for audio_path in tqdm(speakers_audios_names[speaker], desc=f"Extracting audios for speaker: {speaker}"):
            silence_cleaned_audio, sr = utl.remove_silence(audio_path)

            # MFCCs handling
            mfccs = extract_mfccs(silence_cleaned_audio, sr)
            speaker_audios_mfccs[speaker].append(mfccs)

            # LPCCs handling
            lpccs = extract_lpccs(silence_cleaned_audio, sr)
            speaker_audios_lpccs[speaker].append(lpccs)

            # Mel-scaled log-spectrogram handling
            mel_spectrum = extract_mel_spectrum(silence_cleaned_audio, sr, n_filter_bank=MEL_FILTER_BANK_DEFAULT)
            speaker_audios_mel_spectrogram[speaker].append(mel_spectrum)

            # Update max frame num if found higher number of frames
            if len(mfccs) > max_frames_mfcc:
                max_frames_mfcc = len(mfccs)

            if len(lpccs) > max_frames_lpcc:
                max_frames_lpcc = len(lpccs)

            if len(mel_spectrum) > max_frames_mel:
                max_frames_mel = len(mel_spectrum)

    return speaker_audios_mfccs, speaker_audios_lpccs, speaker_audios_mel_spectrogram, max_frames_mfcc,\
        max_frames_lpcc, max_frames_mel


def _fill_speakers_audios_features(speaker_audio_features: dict, max_frames: int, feature_num: int = 0,
                                   mode: int = 0) -> dict:
    """
     Fills each given audio frame array of the input dictionary either with 0s or repeating the frames circularly.

    :param speaker_audio_features: A dictionary of speaker-MFCCs/LPCCs pairs
    :param max_frames: An integer. The target length to normalize each audio
    :param feature_num: An integer. Number of features calculated for each frame. If given 0, it will be inferred
                        automatically from the length first frame of the first audio (assuming all the frames are the
                        same length).
    :param mode: An integer, either 0 or 1, if 0 each frame will be filled with 0-valued frames, if 1 it will be
                 filled repeating audio frames in a circular fashion.
    :return: A normalized dictionary of speaker-MFCCs/LPCCs pairs
    """
    speaker_audios_features_filled = {}

    # Generate desc string
    mode_string = ""
    if mode == 0:
        mode_string = "with 0-valued frames padding"
    elif mode == 1:
        mode_string = "repeating frames in a circular fashion"
    desc = f"Filling audio {mode_string} with max_frames: {max_frames}, feature_num: {feature_num}"

    # For each speaker
    for speaker in tqdm(speaker_audio_features, desc=desc):
        # If given feature_num is 0, infer feature number by looking at the first audio frame length
        if feature_num == 0:
            feature_num = len(speaker_audio_features[speaker][0][0])

        # Create empty numpy _AUDIO_PER_SPEAKER x max_frames x feature_num tensor to contain the filled audio features
        speaker_audios_features_filled[speaker] = np.zeros(
            shape=(1, max_frames, feature_num),
            dtype=np.float64
        )

        # For each audio of the speaker
        for speaker_audio in speaker_audio_features[speaker]:
            # Fill with zero-value features or in a circular fashion based on the given mode
            filled_audio = utl.fill_audio_frames(speaker_audio, target_len=max_frames, mode=mode)
            speaker_audios_features_filled[speaker] = np.append(
                speaker_audios_features_filled[speaker],
                np.reshape(filled_audio, newshape=(1, max_frames, feature_num)),
                axis=0
            )

        # Remove first, empty matrix from the result tensor
        speaker_audios_features_filled[speaker] = speaker_audios_features_filled[speaker][1:]

    return speaker_audios_features_filled


def _generate_speakers_acoustic_model(speakers_audios_features: dict, n_states: int, n_mix: int) -> (dict, dict):
    """
    Generates a trained GMM-HMM model representing the speaker's audio for each speaker's audio and stores it in a
    dictionary of speaker-acoustic_models pairs, a list containing the viterbi-calculated most likely state sequence
    for each audio x in X (i.e. GMM-HMM state sequence y that maximizes P(y | x))audio in X and stores it in a
    dictionary of speaker-acoustic_models_states pairs for each speaker's audio.

    :param speakers_audios_features:  A dictionary of speaker-MFCCs/LPCCs/Mel-spectrogram pairs.
    :param n_states: number of states to generate the acoustic model.
    :param n_mix: number of mixtures for each state.
    :return: A dictionary of speaker-acoustic_models pairs and a dictionary of speaker-acoustic_models_states pairs
    """
    acoustic_models = {}
    acoustic_model_state_labels = {}

    desc = f"Generating speaker-acoustic_models with n_states: {n_states}, n_mix: {n_mix}"
    # For each speaker
    for speaker in tqdm(speakers_audios_features, desc=desc):
        print(f"\nCurrent speaker: {str(speaker)}")
        # Flatten feature matrix into array of frame features
        speaker_audios = speakers_audios_features[speaker]

        # Extract acoustic models and frame-level labels (most likely sequence of states from viterbi algorithm)
        acoustic_models[speaker], acoustic_model_state_labels[speaker] = generate_acoustic_model(
            speaker_audios,
            label=speaker,
            n_states=n_states,
            n_mix=n_mix
        )

        path = f"{ACOUSTIC_MODEL_PATH}{speaker}.pkl"
        save_acoustic_model(acoustic_models[speaker], path)

    return acoustic_models, acoustic_model_state_labels


def _one_hot_encode_state_labels(speakers_raw_state_labels: dict, speaker_indexes: dict, n_states: int) -> \
        list[sp.lil_matrix]:
    speakers_global_state_labels = {}
    n_audio = 0  # audio number counter
    max_frames = 0  # maximum number of frames

    desc = f"Generating one-hot encode state labels with for audio features with n_states: {n_states}"
    # For each speaker
    for speaker in tqdm(speaker_indexes, desc=desc):
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
                global_state_label = (n_states * speaker_indexes[speaker]) + raw_state_label
                global_audio_state_labels = np.append(global_audio_state_labels, global_state_label)

            speakers_global_state_labels[speaker].append(global_audio_state_labels)

    n_speaker = len(speakers_global_state_labels)

    # Create a list to contain the n_audio sparse matrices representing the one-hot encoding of the state labels with
    # shape max_frames x (n_states*n_speaker)
    one_hot_encoded_state_labels = []

    # For each speaker
    for speaker in speaker_indexes:

        # For each audio of the speaker
        for global_audio_state_labels in speakers_global_state_labels[speaker]:

            # Create max_frames x (n_states*n_speaker) matrix representing the one-hot encoding of the state labels
            speaker_one_hot_encoded_state_labels = np.zeros(shape=(max_frames, n_states * n_speaker))

            # For each frame of the audio
            for frame_index in range(0, len(global_audio_state_labels)):

                # Get the target most likely state for the frame according to the viterbi algorithm
                state_index = int(global_audio_state_labels[frame_index])

                # Set the corresponding component of the one-hot encode label vector to 1
                speaker_one_hot_encoded_state_labels[frame_index, state_index] = 1

            # Convert the generated one-hot encoded state labels matrix to sparse lil format and store it in the list
            speaker_one_hot_encoded_state_labels = sp.lil_matrix(speaker_one_hot_encoded_state_labels)
            one_hot_encoded_state_labels.append(speaker_one_hot_encoded_state_labels)

    return one_hot_encoded_state_labels


def _generate_audios_feature_tensor(speaker_audios_features: dict, speaker_indexes: dict) -> np.ndarray:
    # Create n_audio x max_frames x (n_features) tensor to contain the feature matrix of each audio
    audios_feature_tensor = None

    # For each speaker
    for speaker in tqdm(speaker_indexes, desc="Generating audios feature tensor"):
        # Get the feature matrix for each audio of the speaker, and concatenate it to the final feature tensor
        speaker_audios_feature_tensor = speaker_audios_features[speaker]

        # If output tensor is still empty, then copy the feature speaker's audios feature tensor into output tensor,
        # otherwise concatenate it to the current output tensor
        if audios_feature_tensor is None:
            audios_feature_tensor = np.copy(speaker_audios_feature_tensor)
        else:
            audios_feature_tensor = np.concatenate((audios_feature_tensor, speaker_audios_feature_tensor), axis=0)

    return audios_feature_tensor


def _generate_output_dataframe(audios_feature_tensor: np.ndarray, one_hot_encoded_labels: list[sp.lil_matrix]) -> \
        pd.DataFrame:
    """

    :rtype: pd.DataFrame
    """
    df = pd.DataFrame(columns=[AUDIO_DATAFRAME_KEY, STATE_PROB_KEY])
    df[AUDIO_DATAFRAME_KEY] = df[AUDIO_DATAFRAME_KEY].astype(object)
    df[STATE_PROB_KEY] = df[STATE_PROB_KEY].astype(object)

    # For each audio feature matrix
    for i in tqdm(range(0, len(audios_feature_tensor)), desc="Generating output dataframes"):
        # Get feature matrix and state labels probabilities, putting them into a pandas dataframe
        df.loc[i] = (audios_feature_tensor[i], one_hot_encoded_labels[i])

    return df


def main():

    # Get audio paths, grouped by speaker
    speakers_audios_names = speaker_audio_filenames(
        path=DATASET_ORIGINAL_PATH,
        speaker_dir_regex=SPEAKER_DIR_REGEX,
        audio_file_regex=AUDIO_REGEX
    )

    # Generate the speaker indexes to ensure the speakers are always processed in the same order (or load it if saved)
    speaker_indexes = generate_or_load_speaker_ordered_dict(list(speakers_audios_names.keys()), generate=True)

    # Get max frame len, audio mel-scaled spectrograms, audio MFCCs and LPCCs for each speaker
    speaker_audios_mfccs, speaker_audios_lpccs, speaker_audios_mel_spectrograms, max_frames_mfcc, max_frames_lpcc, \
        max_frames_mel = _speakers_audios_mfccs_lpccs_mel_spectrograms_max_frames(speakers_audios_names)

    # Fill speaker audios in a circular fashion up to the max frame number
    speaker_audios_mfcc_filled_circular = _fill_speakers_audios_features(
        speaker_audios_mfccs,
        max_frames_mfcc,
        feature_num=MFCC_NUM_DEFAULT * (DERIVATIVE_ORDER_DEFAULT + 1),
        mode=1
    )
    speaker_audios_lpcc_filled_circular = _fill_speakers_audios_features(
        speaker_audios_lpccs,
        max_frames=max_frames_lpcc,
        feature_num=LPCC_NUM_DEFAULT,
        mode=1
    )
    speaker_audios_mel_spectrogram_filled_circular = _fill_speakers_audios_features(
        speaker_audios_mel_spectrograms,
        max_frames=max_frames_mel,
        feature_num=MEL_FILTER_BANK_DEFAULT,
        mode=1
    )

    # Construct acoustic models and extract frame-level labels for each variation of the features (MFCCs, LPCCs,
    # Mel-scaled spectrograms)
    acoustic_models_mel_spectr_filled_circular, labels_mel_spectr_filled_circular = _generate_speakers_acoustic_model(
        speaker_audios_mel_spectrogram_filled_circular,
        n_states=N_STATES_MEL_SPEC,
        n_mix=N_MIX_MEL_SPEC
    )
    acoustic_models_mfcc_filled_circular, labels_mfcc_filled_circular = _generate_speakers_acoustic_model(
        speaker_audios_mfcc_filled_circular,
        n_states=N_STATES_MFCCS,
        n_mix=N_MIX_MFCCS
    )
    acoustic_models_lpcc_filled_circular, labels_lpcc_filled_circular = _generate_speakers_acoustic_model(
        speaker_audios_lpcc_filled_circular,
        n_states=N_STATES_LPCCS,
        n_mix=N_MIX_LPCCS
    )

    # One-hot encode frame-level state labels as vectors
    one_hot_encoded_labels_mfcc_filled_circular = _one_hot_encode_state_labels(
        speakers_raw_state_labels=labels_mfcc_filled_circular,
        speaker_indexes=speaker_indexes,
        n_states=N_STATES_MFCCS
    )
    one_hot_encoded_labels_lpcc_filled_circular = _one_hot_encode_state_labels(
        speakers_raw_state_labels=labels_lpcc_filled_circular,
        speaker_indexes=speaker_indexes,
        n_states=N_STATES_LPCCS
    )
    one_hot_encoded_labels_mel_spectrogram_filled_circular = _one_hot_encode_state_labels(
        speakers_raw_state_labels=labels_mel_spectr_filled_circular,
        speaker_indexes=speaker_indexes,
        n_states=N_STATES_MEL_SPEC
    )

    # Construct the audio feature tensor for both MFCCs and LPCCs
    audios_feature_tensor_mfcc_filled_circular = _generate_audios_feature_tensor(
        speaker_audios_mfcc_filled_circular,
        speaker_indexes
    )
    audios_feature_tensor_lpcc_filled_circular = _generate_audios_feature_tensor(
        speaker_audios_lpcc_filled_circular,
        speaker_indexes
    )
    audios_feature_tensor_mel_spectrogram_filled_circular = _generate_audios_feature_tensor(
        speaker_audios_mel_spectrogram_filled_circular,
        speaker_indexes
    )

    # Generate dataframes containing input (feature) and output (label) tensors
    df_mfcc_filled_circular = _generate_output_dataframe(
        audios_feature_tensor_mfcc_filled_circular,
        one_hot_encoded_labels_mfcc_filled_circular
    )
    df_lpcc_filled_circular = _generate_output_dataframe(
        audios_feature_tensor_lpcc_filled_circular,
        one_hot_encoded_labels_lpcc_filled_circular
    )
    df_mel_spectrogram_filled_circular = _generate_output_dataframe(
        audios_feature_tensor_mel_spectrogram_filled_circular,
        one_hot_encoded_labels_mel_spectrogram_filled_circular
    )

    # Split generated dataframes into train and test sets
    df_mfcc_filled_circular_train, df_mfcc_filled_circular_test = skl.model_selection.train_test_split(
        df_mfcc_filled_circular,
        train_size=TRAIN_PERCENTAGE,
        shuffle=True,
        random_state=_RANDOM_SEED
    )
    df_lpcc_filled_circular_train, df_lpcc_filled_circular_test = skl.model_selection.train_test_split(
        df_lpcc_filled_circular,
        train_size=TRAIN_PERCENTAGE,
        shuffle=True,
        random_state=_RANDOM_SEED
    )
    df_mel_spectr_filled_circular_train, df_mel_spectr_filled_circular_test = skl.model_selection.train_test_split(
        df_mel_spectrogram_filled_circular,
        train_size=TRAIN_PERCENTAGE,
        shuffle=True,
        random_state=_RANDOM_SEED
    )

    # Save extracted features and labels to pickle files in a suitable format for model training
    df_mfcc_filled_circular_train.to_pickle(TRAIN_SET_PATH_MFCCS)
    df_mfcc_filled_circular_test.to_pickle(TEST_SET_PATH_MFCCS)

    df_lpcc_filled_circular_train.to_pickle(TRAIN_SET_PATH_LPCCS)
    df_lpcc_filled_circular_test.to_pickle(TEST_SET_PATH_LPCCS)

    df_mel_spectr_filled_circular_train.to_pickle(TRAIN_SET_PATH_MEL_SPEC)
    df_mel_spectr_filled_circular_test.to_pickle(TEST_SET_PATH_MEL_SPEC)


if __name__ == "__main__":
    main()
