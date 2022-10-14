from typing import final, Optional
from tqdm.auto import tqdm
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn as skl
import scipy.sparse as sp
from preprocessing import utils as utl
from preprocessing.features.mel import extract_mfccs, extract_mel_spectrum, MFCC_NUM_DEFAULT, MEL_FILTER_BANK_DEFAULT, \
    DERIVATIVE_ORDER_DEFAULT
from preprocessing.features.lpcc import extract_lpccs, LPCC_NUM_DEFAULT
from preprocessing.acoustic_model.gmmhmm import generate_acoustic_model, save_acoustic_model
from preprocessing.constants import ACOUSTIC_MODEL_PATH_MFCCS, ACOUSTIC_MODEL_PATH_LPCCS, ACOUSTIC_MODEL_PATH_MEL_SPEC, \
    TRAIN_PERCENTAGE, AUDIO_DATAFRAME_KEY, STATE_PROB_KEY, N_STATES_MFCCS, N_MIX_MFCCS, N_STATES_LPCCS, N_MIX_LPCCS, \
    N_STATES_MEL_SPEC, N_MIX_MEL_SPEC, DATASET_ORIGINAL_PATH, TRAIN_SET_PATH_MFCCS, TRAIN_SET_PATH_LPCCS, \
    TRAIN_SET_PATH_MEL_SPEC, TEST_SET_PATH_MFCCS, TEST_SET_PATH_LPCCS, TEST_SET_PATH_MEL_SPEC, AUDIO_PER_SPEAKER, \
    SPEAKER_DATAFRAME_KEY, AUDIO_NAME_DATAFRAME_KEY, TEST_PERCENTAGE, VALIDATION_PERCENTAGE, RANDOM_SEED, \
    MAX_FRAMES_MFCCS, MAX_FRAMES_MEL_SPEC, MAX_FRAMES_LPCCS, AUTOTUNE, MAX_FRAMES_WAVEFORM, SAMPLE_RATE_DATAFRAME_KEY
from preprocessing.file_utils import speaker_audio_filenames, generate_or_load_speaker_ordered_dict, \
    SPEAKER_DIR_REGEX, AUDIO_REGEX


_RANDOM_SEED: final = 47


def _create_filename_df(speakers_audios_names: dict, speaker_indexes: dict) -> pd.DataFrame:
    """
    Takes a dictionary of speakers and their audio filenames and returns a dataframe with two columns: one for the audio
    filenames and one for the corresponding speaker.

    :param speakers_audios_names: a dictionary containing the speakers as keys and the audio filenames as values
    :type speakers_audios_names: dict
    :param speaker_indexes: a dictionary mapping each speaker into an integer indicating its index.
    :type speaker_indexes: dict

    :return: A dataframe containing the audio filenames and the corresponding speaker.
    """

    df = pd.DataFrame(columns=[AUDIO_NAME_DATAFRAME_KEY, SPEAKER_DATAFRAME_KEY])
    count = 0

    # For each speaker
    for speaker in speaker_indexes:

        # For each audio filename of the speaker
        for audioname in speakers_audios_names[speaker]:

            # Add a dataframe entry containing audio filename and corresponding speaker
            df.loc[count] = (audioname, speaker)
            count += 1

    return df


def _train_validation_test_split(filename_df: pd.DataFrame, speaker_indexes: dict) -> (pd.DataFrame, pd.DataFrame,
                                                                                       pd.DataFrame):
    """
    Takes a dataframe with the audio names and the corresponding speaker, and returns three dataframes with the
    train, validation and test audio names and speakers.

    :param filename_df: the dataframe containing the filenames and the corresponding speakers
    :type filename_df: pd.DataFrame
    :param speaker_indexes: dict
    :type speaker_indexes: dict

    :return: A tuple of 3 dataframes, one for each of the train, validation and test sets.
    """

    df_train = pd.DataFrame(columns=[AUDIO_NAME_DATAFRAME_KEY, SPEAKER_DATAFRAME_KEY])
    df_validation = pd.DataFrame(columns=[AUDIO_NAME_DATAFRAME_KEY, SPEAKER_DATAFRAME_KEY])
    df_test = pd.DataFrame(columns=[AUDIO_NAME_DATAFRAME_KEY, SPEAKER_DATAFRAME_KEY])

    # For each speaker
    for speaker in speaker_indexes:

        # Get the sub-dataframe corresponding to the speaker
        speaker_df = filename_df[filename_df[SPEAKER_DATAFRAME_KEY] == speaker]

        # Perform train/validation/test split
        speaker_df_train, speaker_df_test = skl.model_selection.train_test_split(
            speaker_df,
            test_size=TEST_PERCENTAGE,
            random_state=RANDOM_SEED
        )
        speaker_df_train, speaker_df_validation = skl.model_selection.train_test_split(
            speaker_df_train,
            test_size=VALIDATION_PERCENTAGE / (1 - TEST_PERCENTAGE),
            random_state=RANDOM_SEED
        )

        # Add result to final dataframes
        df_train = pd.concat([df_train, speaker_df_train], axis=0)
        df_test = pd.concat([df_test, speaker_df_test], axis=0)
        df_validation = pd.concat([df_validation, speaker_df_validation], axis=0)

    return df_train, df_validation, df_test


def get_feature_waveform(audio_path, speaker):
    features_waveform = tf.numpy_function(
        __get_feature_waveform_func,
        [audio_path, speaker],
        [tf.float32, tf.int32, tf.string, tf.string]
    )
    features_waveform[0].set_shape(
        tf.TensorShape((MAX_FRAMES_WAVEFORM, ))
    )

    # Cast path and speaker id to tf.string and sr to tf.int32
    features_waveform[1].set_shape(tf.TensorShape(()))
    features_waveform[2].set_shape(tf.TensorShape(()))
    features_waveform[3].set_shape(tf.TensorShape(()))

    tf.cast(features_waveform[1], tf.int32, name=SAMPLE_RATE_DATAFRAME_KEY)
    tf.cast(features_waveform[2], tf.string, name=AUDIO_NAME_DATAFRAME_KEY)
    tf.cast(features_waveform[3], tf.string, name=SPEAKER_DATAFRAME_KEY)

    return features_waveform


def __get_feature_waveform_func(audio_path, speaker):

    _audio_path = audio_path.decode('UTF-8')

    # Extract waveform removing silence frames
    waveform, sr = utl.remove_silence(str(_audio_path))

    # Pad the sequence with repeating frames
    filled_audio = utl.fill_audio_frames(
        waveform,
        target_len=MAX_FRAMES_WAVEFORM,
        mode=utl.REPEATING_FRAMES_PADDING_MODE
    )

    return filled_audio.astype(np.float32), sr, audio_path, speaker


def get_feature_mfccs(waveform, sr, audio_path, speaker):
    features_mfccs = tf.numpy_function(
        __get_feature_mfccs_func,
        [waveform, sr, audio_path, speaker],
        [tf.float32, tf.int32, tf.string, tf.string]
    )

    # Adjust shapes which are lost in tf.numpy_function() execution
    features_mfccs[0].set_shape(
        tf.TensorShape([MAX_FRAMES_MFCCS, MFCC_NUM_DEFAULT*(DERIVATIVE_ORDER_DEFAULT + 1)])
    )

    # Cast path and speaker id to tf.string and sr to tf.int32
    features_mfccs[1].set_shape(tf.TensorShape(()))
    features_mfccs[2].set_shape(tf.TensorShape(()))
    features_mfccs[3].set_shape(tf.TensorShape(()))

    tf.cast(features_mfccs[1], tf.int32, name=SAMPLE_RATE_DATAFRAME_KEY)
    tf.cast(features_mfccs[2], tf.string, name=AUDIO_NAME_DATAFRAME_KEY)
    tf.cast(features_mfccs[3], tf.string, name=SPEAKER_DATAFRAME_KEY)

    return features_mfccs


def __get_feature_mfccs_func(waveform, sr, audio_path, speaker):

    # Extract MFCCs
    mfccs = extract_mfccs(waveform, sr, n_mfcc=MFCC_NUM_DEFAULT, order=DERIVATIVE_ORDER_DEFAULT)

    # Pad the sequence with repeating frames (not necessary anymore since padding is done while extracting waveforms)
    # filled_audio = utl.fill_audio_frames(mfccs, target_len=MAX_FRAMES_MFCCS, mode=utl.REPEATING_FRAMES_PADDING_MODE)

    return mfccs.astype(np.float32), sr, audio_path, speaker


def get_feature_mel_spec(x, speaker):
    features_mel_spec = tf.numpy_function(
        __get_feature_mel_spec_func,
        [x, speaker],
        [tf.float32, tf.string, tf.string]
    )

    # Adjust shapes which are lost in tf.numpy_function() execution
    features_mel_spec[0].set_shape(
        tf.TensorShape([MAX_FRAMES_MEL_SPEC, MEL_FILTER_BANK_DEFAULT])
    )

    # Cast path and speaker id to tf.string
    features_mel_spec[1].set_shape(tf.TensorShape(()))
    features_mel_spec[2].set_shape(tf.TensorShape(()))
    tf.cast(features_mel_spec[1], tf.string, name=AUDIO_NAME_DATAFRAME_KEY)
    tf.cast(features_mel_spec[2], tf.string, name=SPEAKER_DATAFRAME_KEY)

    return features_mel_spec


def __get_feature_mel_spec_func(audio_path, speaker):

    _audio_path = audio_path.decode('UTF-8')

    silence_cleaned_audio, sr = utl.remove_silence(str(_audio_path))

    # Extract Mel-Scaled log-spectrum
    mel_spec = extract_mel_spectrum(silence_cleaned_audio, sr, n_filter_bank=MEL_FILTER_BANK_DEFAULT)

    # Pad the sequence with repeating frames
    filled_audio = utl.fill_audio_frames(
        mel_spec,
        target_len=MAX_FRAMES_MEL_SPEC,
        mode=utl.REPEATING_FRAMES_PADDING_MODE
    )

    return filled_audio.astype(np.float32), audio_path, speaker


def get_feature_lpccs(x, speaker):
    features_lpccs = tf.numpy_function(
        __get_feature_lpccs,
        [x, speaker],
        [tf.float32, tf.string, tf.string]
    )

    # Adjust shapes which are lost in tf.numpy_function() execution
    features_lpccs[0].set_shape(
        tf.TensorShape([MAX_FRAMES_LPCCS, LPCC_NUM_DEFAULT])
    )

    # Cast path and speaker id to tf.string
    features_lpccs[1].set_shape(tf.TensorShape(()))
    features_lpccs[2].set_shape(tf.TensorShape(()))
    tf.cast(features_lpccs[1], tf.string, name=AUDIO_NAME_DATAFRAME_KEY)
    tf.cast(features_lpccs[2], tf.string, name=SPEAKER_DATAFRAME_KEY)

    return features_lpccs


def __get_feature_lpccs(audio_path, speaker):

    _audio_path = audio_path.decode('UTF-8')

    silence_cleaned_audio, sr = utl.remove_silence(_audio_path)

    # Extract LPCCs
    lpccs = extract_lpccs(silence_cleaned_audio, sr, n_lpcc=LPCC_NUM_DEFAULT)

    # Pad the sequence with repeating frames
    filled_audio = utl.fill_audio_frames(lpccs, target_len=MAX_FRAMES_LPCCS, mode=utl.REPEATING_FRAMES_PADDING_MODE)

    return filled_audio.astype(np.float32), audio_path, speaker


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


def _generate_audios_feature_tensor(speaker_audios_features: dict, speaker_indexes: dict) -> np.ndarray:
    """
    Generates a tensor containing a feature matrix for each audio.

    :param speaker_audios_features: dictionary of speaker-audio features (MFCCs, LPCCs, Mel-scaled log-spectrogram)
        pairs.
    :param speaker_indexes: a dictionary indicating the processing order for the speakers.
    :return: a (n_audio, max_frames, n_features)-shaped tensor, where each of the n_audio matrices contains the audio
        features for each frame.
    """
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
    Converts the (n_audio, max_frames, n_features)-shaped and the list containing the one-hot encoding of the
    frame-level state labels into a pandas dataframe.

    :param audios_feature_tensor: (n_audio, max_frames, n_features)-shaped tensor, where each of the n_audio matrices
        contains the audio features for each frame.
    :param one_hot_encoded_labels: list of scipy sparse matrices each containing the one-hot encoding of frame-level
        state labels for a single audio.
    :return: a pandas DataFrame object with 2 columns: the first one containing the audios_feature_tensor entries on
        each row, and the second one containing one_hot_encoded_labels entries on each row.
    """
    df = pd.DataFrame(columns=[AUDIO_DATAFRAME_KEY, STATE_PROB_KEY])
    df[AUDIO_DATAFRAME_KEY] = df[AUDIO_DATAFRAME_KEY].astype(object)
    df[STATE_PROB_KEY] = df[STATE_PROB_KEY].astype(object)

    # For each audio feature matrix
    for i in tqdm(range(0, len(audios_feature_tensor)), desc="Generating output dataframes"):
        # Get feature matrix and state labels probabilities, putting them into a pandas dataframe
        df.loc[i] = (audios_feature_tensor[i], one_hot_encoded_labels[i])

    return df


def _train_test_split(feature_dataframe: pd.DataFrame, train_percentage: float, audios_per_speaker: int) \
        -> (pd.DataFrame, pd.DataFrame):
    """
    Splits feature dataframe into train and test set.

    :param feature_dataframe: a pandas DataFrame object with 2 columns: the first one containing the
        audios_feature_tensor entries on each row, and the second one containing one_hot_encoded_labels entries on each
        row.
    :param audios_per_speaker: number of audios per speaker.
    :param train_percentage: percentage of samples to put in the train test.
    :return: train/test-splitted feature dataframe, where train set contains percentage*audios_per_speaker audios from
        each speaker, while the test set (1-train_percentage)*audios_per_speaker audios from each speaker.
    """
    columns = list(feature_dataframe.columns)
    train = pd.DataFrame(columns=columns)
    test = pd.DataFrame(columns=columns)

    # Convert dataframes into object type
    for column in columns:
        train[column] = train[column].astype(object)
        test[column] = test[column].astype(object)

    n_speaker = feature_dataframe.shape[0] // audios_per_speaker  # must give integer value

    # For each speaker
    for i in tqdm(range(0, n_speaker), desc="Executing train/test split"):

        # Get speaker audios
        speaker_audios = feature_dataframe.loc[i*audios_per_speaker:i*audios_per_speaker + audios_per_speaker]

        # Split speaker audios in train/test set
        train_speaker, test_speaker = skl.model_selection.train_test_split(
            speaker_audios,
            train_size=train_percentage,
            shuffle=True,
            random_state=_RANDOM_SEED
        )

        # Concatenate speaker train/test sets to global train/test sets
        train = pd.concat([train, train_speaker], axis=0)
        test = pd.concat([test, test_speaker], axis=0)

    # Shuffle the generated train/test sets
    train = skl.utils.shuffle(train, random_state=_RANDOM_SEED)
    test = skl.utils.shuffle(test, random_state=_RANDOM_SEED)

    return train, test


# TODO: fix train/test split
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
    filename_df = _create_filename_df(speakers_audios_names, speaker_indexes)
    df_train, df_val, df_test = _train_validation_test_split(filename_df, speaker_indexes)

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

    # Gret audio features
    train_prebatch_mfccs = train_prebatch_waveform.map(get_feature_mfccs, num_parallel_calls=AUTOTUNE)
    train_prebatch_lpccs = train_prebatch_waveform.map(get_feature_lpccs, num_parallel_calls=AUTOTUNE)
    train_prebatch_mel_spec = train_prebatch_waveform.map(get_feature_mel_spec, num_parallel_calls=AUTOTUNE)

    val_prebatch_mfccs = val_prebatch_waveform.map(get_feature_mfccs, num_parallel_calls=AUTOTUNE)
    val_prebatch_lpccs = val_prebatch_waveform.map(get_feature_lpccs, num_parallel_calls=AUTOTUNE)
    val_prebatch_mel_spec = val_prebatch_waveform.map(get_feature_mel_spec, num_parallel_calls=AUTOTUNE)

    test_prebatch_mfccs = test_prebatch_waveform.map(get_feature_mfccs, num_parallel_calls=AUTOTUNE)
    test_prebatch_lpccs = test_prebatch_waveform.map(get_feature_lpccs, num_parallel_calls=AUTOTUNE)
    test_prebatch_mel_spec = test_prebatch_waveform.map(get_feature_mel_spec, num_parallel_calls=AUTOTUNE)


if __name__ == "__main__":
    main()
