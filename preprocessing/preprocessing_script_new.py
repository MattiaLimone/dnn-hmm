from typing import Optional
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tqdm.auto import tqdm
from preprocessing.acoustic_model.gmmhmm import generate_acoustic_model, save_acoustic_model
from preprocessing.constants import DATASET_ORIGINAL_PATH, SPEAKER_DATAFRAME_KEY, \
    AUDIO_NAME_DATAFRAME_KEY, AUTOTUNE
from preprocessing.dataset_transformations import create_filename_df, train_validation_test_split, \
    get_feature_waveform, get_feature_mfccs, get_feature_mel_spec, get_feature_lpccs
from preprocessing.file_utils import speaker_audio_filenames, generate_or_load_speaker_ordered_dict, \
    SPEAKER_DIR_REGEX, AUDIO_REGEX


def _generate_speakers_acoustic_model(speakers_audios_features: dict, n_states: int, n_mix: int,
                                      export_path: Optional[str] = None) -> (dict, dict):
    """
    Generates a trained GMM-HMM model representing the speaker's audio for each speaker's audio and stores it in a
    dictionary of speaker-acoustic_models pairs, a list containing the viterbi-calculated most likely state sequence
    for each audio x in X (i.e. GMM-HMM state sequence y that maximizes P(y | x))audio in X and stores it in a
    dictionary of speaker-acoustic_models_states pairs for each speaker's audio.

    :param speakers_audios_features:  A dictionary of speaker-MFCCs/LPCCs/Mel-spectrogram pairs.
    :param n_states: number of states to generate the acoustic model.
    :param n_mix: number of mixtures for each state.
    :param export_path: path to save the acoustic model into (by default this is None, meaning no export file will be
        generated).
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

        if export_path is not None:
            path = f"{export_path}{speaker}.pkl"
            save_acoustic_model(acoustic_models[speaker], path)

    return acoustic_models, acoustic_model_state_labels


def _one_hot_encode_state_labels(speakers_raw_state_labels: dict, speaker_indexes: dict, n_states: int) -> \
        list[sp.lil_matrix]:
    """
    Generates the one-hot encoding of the frame-level state labels.

    :param speakers_raw_state_labels: a dictionary of speaker-frame level state labels for each audio pairs.
    :param speaker_indexes: a dictionary indicating the processing order for the speakers.
    :param n_states: number of HMM states for each speaker.
    :return: a list of scipy sparse matrices each containing the one-hot encoding of frame-level state labels for a
        single audio.
    """
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


def main():

    # Get audio paths, grouped by speaker
    speakers_audios_names = speaker_audio_filenames(
        path=DATASET_ORIGINAL_PATH,
        speaker_dir_regex=SPEAKER_DIR_REGEX,
        audio_file_regex=AUDIO_REGEX
    )

    # Generate the speaker indexes to ensure the speakers are always processed in the same order (or load it if saved)
    speaker_indexes = generate_or_load_speaker_ordered_dict(list(speakers_audios_names.keys()), generate=True)

    # Generate dataframes for train/test/validation sets
    filename_df = create_filename_df(speakers_audios_names, speaker_indexes)
    df_train, df_val, df_test = train_validation_test_split(filename_df, speaker_indexes)

    # Convert into tensorflow dataset
    train_prebatch = tf.data.Dataset.from_tensor_slices(
        (df_train[AUDIO_NAME_DATAFRAME_KEY], df_train[SPEAKER_DATAFRAME_KEY])
    )
    val_prebatch = tf.data.Dataset.from_tensor_slices(
        (df_val[AUDIO_NAME_DATAFRAME_KEY], df_val[SPEAKER_DATAFRAME_KEY])
    )
    test_prebatch = tf.data.Dataset.from_tensor_slices(
        (df_test[AUDIO_NAME_DATAFRAME_KEY], df_test[SPEAKER_DATAFRAME_KEY])
    )

    # Get audio waveforms
    train_prebatch_waveform = train_prebatch.map(get_feature_waveform, num_parallel_calls=AUTOTUNE)
    val_prebatch_waveform = val_prebatch.map(get_feature_waveform, num_parallel_calls=AUTOTUNE)
    test_prebatch_waveform = test_prebatch.map(get_feature_waveform, num_parallel_calls=AUTOTUNE)

    # TODO: perform data augmentation on training data

    # Get audio features
    train_prebatch_mfccs = train_prebatch_waveform.map(get_feature_mfccs, num_parallel_calls=AUTOTUNE)
    train_prebatch_lpccs = train_prebatch_waveform.map(get_feature_lpccs, num_parallel_calls=AUTOTUNE)
    train_prebatch_mel_spec = train_prebatch_waveform.map(get_feature_mel_spec, num_parallel_calls=AUTOTUNE)

    val_prebatch_mfccs = val_prebatch_waveform.map(get_feature_mfccs, num_parallel_calls=AUTOTUNE)
    val_prebatch_lpccs = val_prebatch_waveform.map(get_feature_lpccs, num_parallel_calls=AUTOTUNE)
    val_prebatch_mel_spec = val_prebatch_waveform.map(get_feature_mel_spec, num_parallel_calls=AUTOTUNE)

    test_prebatch_mfccs = test_prebatch_waveform.map(get_feature_mfccs, num_parallel_calls=AUTOTUNE)
    test_prebatch_lpccs = test_prebatch_waveform.map(get_feature_lpccs, num_parallel_calls=AUTOTUNE)
    test_prebatch_mel_spec = test_prebatch_waveform.map(get_feature_mel_spec, num_parallel_calls=AUTOTUNE)

    # TODO: generate GMM-HMM acoustic models from training data (for each speaker)

    # TODO: extract frame-level state labels applying Viterbi algorithm on each audio and the speaker's acoustic model

    # TODO: (maybe) one-hot encode the state labels

    # TODO: save output datasets


if __name__ == "__main__":
    main()
