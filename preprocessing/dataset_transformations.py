import os
import numpy as np
import pandas as pd
import sklearn as skl
import tensorflow as tf
from preprocessing import utils as utl
from preprocessing.constants import AUDIO_NAME_DATAFRAME_KEY, SPEAKER_DATAFRAME_KEY, TEST_PERCENTAGE, RANDOM_SEED, \
    VALIDATION_PERCENTAGE, MAX_FRAMES_WAVEFORM, SAMPLE_RATE_DATAFRAME_KEY, MAX_FRAMES_MFCCS, MAX_FRAMES_MEL_SPEC, \
    MAX_FRAMES_LPCCS, ACOUSTIC_MODEL_PATH_MFCCS, N_STATES_MFCCS
from preprocessing.features.lpcc import LPCC_NUM_DEFAULT, extract_lpccs
from preprocessing.features.mel import extract_mfccs, extract_mel_spectrum, MFCC_NUM_DEFAULT, \
    DERIVATIVE_ORDER_DEFAULT, MEL_FILTER_BANK_DEFAULT
from preprocessing.acoustic_model.gmmhmm import load_acoustic_model
from preprocessing.file_utils import generate_or_load_speaker_ordered_dict


def create_filename_df(speakers_audios_names: dict, speaker_indexes: dict) -> pd.DataFrame:
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


def train_validation_test_split(filename_df: pd.DataFrame, speaker_indexes: dict) -> (pd.DataFrame, pd.DataFrame,
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
    """
    It takes in an audio filename and a speaker id, and returns a tuple composed by waveform, sample rate, audio
    name, and speaker id.

    :param audio_path: The path to the audio file
    :param speaker: The speaker id of the audio file
    :return: A tuple of 4 elements:
        1. A tensor of shape (MAX_FRAMES_WAVEFORM, ) representing the waveform.
        2. A tensor of shape () representing the sample rate.
        3. A tensor of shape () representing the audio path.
        4. A tensor of shape () representing the speaker id.
    """
    features_waveform = tf.numpy_function(
        __get_feature_waveform_func,
        [audio_path, speaker],
        [tf.float32, tf.int32, tf.string, tf.string]
    )
    features_waveform[0].set_shape(
        tf.TensorShape((MAX_FRAMES_WAVEFORM,))
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
    """
    Takes an audio file path, loads the audio file, removes silence, pads the audio with repeating frames, and
    returns the audio waveform as a numpy array.

    :param audio_path: The path to the audio file.
    :param speaker: The speaker's name.
    :return: The waveform, the sampling rate, the audio path, and the speaker.
    """

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
    """
    Takes a waveform, sample rate, audio path, and speaker id, and returns a tuple of the mfccs, sample rate, audio
    path, and speaker id.

    :param waveform: The audio waveform.
    :param sr: The sample rate of the audio file.
    :param audio_path: The path to the audio file.
    :param speaker: The speaker id.
    :return: A tuple of 4 elements:
        1. A tensor of shape (MAX_FRAMES_MFCCS, MFCC_NUM_DEFAULT*(DERIVATIVE_ORDER_DEFAULT + 1)) representing the MFCCs.
        2. A tensor of shape () representing the sample rate.
        3. A tensor of shape () representing the audio path.
        4. A tensor of shape () representing the speaker id.
    """
    features_mfccs = tf.numpy_function(
        __get_feature_mfccs_func,
        [waveform, sr, audio_path, speaker],
        [tf.float32, tf.int32, tf.string, tf.string]
    )

    # Adjust shapes which are lost in tf.numpy_function() execution
    features_mfccs[0].set_shape(
        tf.TensorShape([MAX_FRAMES_MFCCS, MFCC_NUM_DEFAULT * (DERIVATIVE_ORDER_DEFAULT + 1)])
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
    """
    Takes in a waveform, sampling rate, audio path, and speaker, and returns the MFCCs of the waveform, the sampling
    rate, the audio path, and the speaker.

    :param waveform: The audio waveform.
    :param sr: The sample rate of the audio file.
    :param audio_path: The path to the audio file.
    :param speaker: The speaker id.
    :return: The MFCCs, the sampling rate, the audio path, and the speaker.
    """

    # Extract MFCCs
    mfccs = extract_mfccs(waveform, sr, n_mfcc=MFCC_NUM_DEFAULT, order=DERIVATIVE_ORDER_DEFAULT)

    # Pad the sequence with repeating frames (not necessary anymore since padding is done while extracting waveforms)
    # filled_audio = utl.fill_audio_frames(mfccs, target_len=MAX_FRAMES_MFCCS, mode=utl.REPEATING_FRAMES_PADDING_MODE)

    return mfccs.astype(np.float32), sr, audio_path, speaker


def get_feature_mel_spec(waveform, sr, audio_path, speaker):
    """
    Takes a waveform, sample rate, audio path, and speaker id, and returns a tuple of the mel spectrogram, sample rate,
    audio path, and speaker id.

    :param waveform: The audio waveform.
    :param sr: The sample rate of the audio file.
    :param audio_path: The path to the audio file.
    :param speaker: The speaker id.
    :return: A tuple of 4 elements:
        1. A tensor of shape (MAX_FRAMES_MEL_SPEC, MEL_FILTER_BANK_DEFAULT) representing the log-scaled Mel spectrum.
        2. A tensor of shape () representing the sample rate.
        3. A tensor of shape () representing the audio path.
        4. A tensor of shape () representing the speaker id.
    """
    features_mel_spec = tf.numpy_function(
        __get_feature_mel_spec_func,
        [waveform, sr, audio_path, speaker],
        [tf.float32, tf.int32, tf.string, tf.string]
    )

    # Adjust shapes which are lost in tf.numpy_function() execution
    features_mel_spec[0].set_shape(
        tf.TensorShape([MAX_FRAMES_MEL_SPEC, MEL_FILTER_BANK_DEFAULT])
    )

    # Cast path and speaker id to tf.string and sr to tf.int32
    features_mel_spec[1].set_shape(tf.TensorShape(()))
    features_mel_spec[2].set_shape(tf.TensorShape(()))
    features_mel_spec[3].set_shape(tf.TensorShape(()))

    tf.cast(features_mel_spec[1], tf.int32, name=SAMPLE_RATE_DATAFRAME_KEY)
    tf.cast(features_mel_spec[2], tf.string, name=AUDIO_NAME_DATAFRAME_KEY)
    tf.cast(features_mel_spec[3], tf.string, name=SPEAKER_DATAFRAME_KEY)

    return features_mel_spec


def __get_feature_mel_spec_func(waveform, sr, audio_path, speaker):
    # Extract log-scaled Mel-spectrum
    mel_spec = extract_mel_spectrum(waveform, sr, n_filter_bank=MEL_FILTER_BANK_DEFAULT)

    return mel_spec.astype(np.float32), sr, audio_path, speaker


def get_feature_lpccs(waveform, sr, audio_path, speaker):
    """
    Takes in a waveform, sample rate, audio path, and speaker id, and returns a tuple of the LPCCs, sample rate, audio
    path, and speaker id.

    :param waveform: The audio waveform.
    :param sr: The sample rate of the audio file.
    :param audio_path: The path to the audio file.
    :param speaker: The speaker id.
    :return: A tuple of 4 elements:
        1. A tensor of shape (MAX_FRAMES_LPCCS, LPCC_NUM_DEFAULT) representing the LPCCs.
        2. A tensor of shape () representing the sample rate.
        3. A tensor of shape () representing the audio path.
        4. A tensor of shape () representing the speaker id.
    """
    features_lpccs = tf.numpy_function(
        __get_feature_lpccs_func,
        [waveform, sr, audio_path, speaker],
        [tf.float32, tf.int32, tf.string, tf.string]
    )

    # Adjust shapes which are lost in tf.numpy_function() execution
    features_lpccs[0].set_shape(
        tf.TensorShape([MAX_FRAMES_LPCCS, LPCC_NUM_DEFAULT])
    )

    # Cast path and speaker id to tf.string and sr to tf.int32
    features_lpccs[1].set_shape(tf.TensorShape(()))
    features_lpccs[2].set_shape(tf.TensorShape(()))
    features_lpccs[3].set_shape(tf.TensorShape(()))

    tf.cast(features_lpccs[1], tf.int32, name=SAMPLE_RATE_DATAFRAME_KEY)
    tf.cast(features_lpccs[2], tf.string, name=AUDIO_NAME_DATAFRAME_KEY)
    tf.cast(features_lpccs[3], tf.string, name=SPEAKER_DATAFRAME_KEY)

    return features_lpccs


def __get_feature_lpccs_func(waveform, sr, audio_path, speaker):
    """
    Takes in a waveform, sampling rate, audio path, and speaker, and returns the LPCCs, sampling rate, audio path,
    and speaker.

    :param waveform: The audio waveform
    :param sr: sampling rate
    :param audio_path: The path to the audio file
    :param speaker: The speaker's name
    :return: The LPCCs, the sampling rate, the audio path, and the speaker.
    """

    # Extract LPCCs
    lpccs = extract_lpccs(waveform, sr, n_lpcc=LPCC_NUM_DEFAULT)

    return lpccs.astype(np.float32), sr, audio_path, speaker


def generate_state_labels_mfccs(mfccs, sr, audio_path, speaker):
    """
    It takes in a batch of MFCCs, and returns a batch of MFCCs, sample rates, audio paths, speakers, and state labels.

    :param mfccs: the mfccs of the audio file
    :param sr: sampling rate of the audio
    :param audio_path: The path to the audio file
    :param speaker: The speaker's name
    :return: mfccs, sr, audio_path, speaker, state_labels.
    """

    # Generate labels using acoustic model
    state_labels = tf.numpy_function(
        __generate_state_labels_mfccs,
        inp=[mfccs, speaker],
        Tout=[tf.int32]
    )[0]

    # Restore shape
    state_labels.set_shape(tf.TensorShape((MAX_FRAMES_MFCCS, )))
    tf.cast(state_labels, tf.int32)

    return mfccs, sr, audio_path, speaker, state_labels


def __generate_state_labels_mfccs(mfccs, speaker):
    """
    It takes in a list of MFCCs and a speaker, and returns a list of state labels.

    :param mfccs: the MFCCs of the audio file
    :param speaker: The name of the speaker
    :return: The state labels for the given mfccs and speaker.
    """
    _speaker = speaker.decode('UTF-8')

    # Load acoustic model
    acoustic_model = load_acoustic_model(f"{os.path.join(ACOUSTIC_MODEL_PATH_MFCCS, _speaker)}.pkl")

    # Load speaker indexes
    speaker_indexes = generate_or_load_speaker_ordered_dict()

    # Generate labels
    log_prob, state_labels = acoustic_model.model.decode(mfccs, algorithm='viterbi')

    for i in range(0, len(state_labels)):
        state_labels[i] = state_labels[i] + (N_STATES_MFCCS * speaker_indexes[_speaker])

    return state_labels.astype(np.int32)




