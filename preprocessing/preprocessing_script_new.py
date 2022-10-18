import numpy as np
import tensorflow as tf
from sequentia import GMMHMM
from tqdm.auto import tqdm
from time import time
from preprocessing.acoustic_model.gmmhmm import generate_acoustic_model, save_acoustic_model
from preprocessing.augmentation.augumentations import augment_waveform_dataset
from preprocessing.constants import DATASET_ORIGINAL_PATH, SPEAKER_DATAFRAME_KEY, \
    AUDIO_NAME_DATAFRAME_KEY, AUTOTUNE, ACOUSTIC_MODEL_PATH_MFCCS, N_STATES_MFCCS, N_MIX_MFCCS, BUFFER_SIZE, \
    TRAIN_WAVEFORMS, TRAIN_SET_PATH_MFCCS_TF
from preprocessing.dataset_transformations import create_filename_df, train_validation_test_split, \
    get_feature_waveform, get_feature_mfccs, get_feature_mel_spec, get_feature_lpccs, generate_state_labels_mfccs
from preprocessing.file_utils import speaker_audio_filenames, generate_or_load_speaker_ordered_dict, \
    SPEAKER_DIR_REGEX, AUDIO_REGEX


def _generate_acoustic_models(feature_dataset: tf.data.Dataset, speaker_indexes: dict, n_states: int, n_mix: int,
                              export_path: str = ACOUSTIC_MODEL_PATH_MFCCS) -> dict[str, GMMHMM]:
    """
    Generates a trained GMM-HMM model representing the speaker's audio for each speaker's audio and stores it in a
    dictionary of speaker-acoustic_models pairs, a list containing the viterbi-calculated most likely state sequence
    for each audio x in X (i.e. GMM-HMM state sequence y that maximizes P(y | x)) audio in X, stores it into a
    dictionary of speaker-acoustic_model pairs for each speaker's audio, storing it into the given directory.

    :param feature_dataset:  A dataset containing the MFCCs/LPCCs/Mel-spectrum features and the corresponding speaker.
    :param speaker_indexes: a dictionary indicating the processing order for the speakers.
    :param n_states: number of states to generate the acoustic model.
    :param n_mix: number of mixtures for each state.
    :param export_path: path to save the acoustic model into.

    :return a dictionary mapping each speaker id into corresponding acoustic model.
    """
    acoustic_models = {}

    for speaker in tqdm(speaker_indexes):

        # Filter speaker sub-dataset
        speaker_ds = feature_dataset.filter(
            lambda features, sr, audio_path, spk: tf.equal(spk, tf.constant(speaker, dtype=tf.string))
        )

        # Create a numpy array containing all the speaker audios
        speaker_audios = []
        for entry in speaker_ds:
            speaker_audios.append(entry[0].numpy())

        speaker_audios = np.array(speaker_audios)

        # Generate acoustic model, ignoring the labels since we'll get them later
        acoustic_models[speaker], acoustic_model_labels = generate_acoustic_model(
            speaker_audios,
            label=speaker,
            n_states=n_states,
            n_mix=n_mix
        )

        # Store the  acoustic model in the give path
        path = f"{export_path}{speaker}.pkl"
        save_acoustic_model(acoustic_models[speaker], path)

    return acoustic_models


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

    # Perform shuffle
    train_prebatch_waveform = train_prebatch_waveform.shuffle(buffer_size=BUFFER_SIZE, reshuffle_each_iteration=True)

    # Try to load waveform to not read all the files again
    try:
        train_prebatch_waveform = tf.data.experimental.load(TRAIN_WAVEFORMS)
        print("Loaded waveforms.")
        for el in train_prebatch_waveform:
            print(el)

    except tf.errors.NotFoundError:
        print("Reading waveforms...")
        tf.data.experimental.save(train_prebatch_waveform, path=TRAIN_WAVEFORMS)
        print("Saved waveforms.")

        train_prebatch_waveform = tf.data.experimental.load(TRAIN_WAVEFORMS)
        print("Loaded waveforms.")

    # Ask user if they want to perform data augmentation on training data
    answer = input("Perform data augmentation (0: no, 1: yes)? ")
    try:
        answer = int(answer)
    except ValueError:
        print("Error: insert 0 for no and 1 for yes.")
    if answer != 0 and answer != 1:
        print("Error: insert 0 for no and 1 for yes.")

    # If the answer is yes, then perform it
    train_prebatch_waveform_no_augment = train_prebatch_waveform
    if answer != 0:
        train_prebatch_waveform = augment_waveform_dataset(train_prebatch_waveform)

        # Perform shuffling
        train_prebatch_waveform = train_prebatch_waveform.shuffle(
            buffer_size=BUFFER_SIZE,
            reshuffle_each_iteration=True
        )

    # Get audio features
    print("Extracting audio features...")
    train_prebatch_mfccs = train_prebatch_waveform.map(get_feature_mfccs, num_parallel_calls=AUTOTUNE)

    # Save the original features if the dataset has been augmented
    train_prebatch_mfccs_no_augment = train_prebatch_mfccs
    if answer != 0:
        train_prebatch_mfccs_no_augment = train_prebatch_waveform_no_augment.map(
            get_feature_mfccs,
            num_parallel_calls=AUTOTUNE
        )

    train_prebatch_lpccs = train_prebatch_waveform.map(get_feature_lpccs, num_parallel_calls=AUTOTUNE)
    train_prebatch_mel_spec = train_prebatch_waveform.map(get_feature_mel_spec, num_parallel_calls=AUTOTUNE)

    val_prebatch_mfccs = val_prebatch_waveform.map(get_feature_mfccs, num_parallel_calls=AUTOTUNE)
    val_prebatch_lpccs = val_prebatch_waveform.map(get_feature_lpccs, num_parallel_calls=AUTOTUNE)
    val_prebatch_mel_spec = val_prebatch_waveform.map(get_feature_mel_spec, num_parallel_calls=AUTOTUNE)

    test_prebatch_mfccs = test_prebatch_waveform.map(get_feature_mfccs, num_parallel_calls=AUTOTUNE)
    test_prebatch_lpccs = test_prebatch_waveform.map(get_feature_lpccs, num_parallel_calls=AUTOTUNE)
    test_prebatch_mel_spec = test_prebatch_waveform.map(get_feature_mel_spec, num_parallel_calls=AUTOTUNE)

    # Perform shuffling
    train_prebatch_mfccs = train_prebatch_mfccs.shuffle(
        buffer_size=BUFFER_SIZE,
        reshuffle_each_iteration=True
    )
    train_prebatch_mfccs_no_augment = train_prebatch_mfccs_no_augment.shuffle(
        buffer_size=BUFFER_SIZE,
        reshuffle_each_iteration=True
    )
    train_prebatch_lpccs = train_prebatch_lpccs.shuffle(
        buffer_size=BUFFER_SIZE,
        reshuffle_each_iteration=True
    )
    train_prebatch_mel_spec = train_prebatch_mel_spec.shuffle(
        buffer_size=BUFFER_SIZE,
        reshuffle_each_iteration=True
    )
    print("Audio features extracted.")

    t0 = time()
    # Generate GMM-HMM acoustic models using MFCCs extracted from training data (for each speaker)
    acoustic_models = _generate_acoustic_models(
        feature_dataset=train_prebatch_mfccs_no_augment,
        speaker_indexes=speaker_indexes,
        n_states=N_STATES_MFCCS,
        n_mix=N_MIX_MFCCS,
        export_path=ACOUSTIC_MODEL_PATH_MFCCS
    )
    t1 = time()
    print(f"Acoustic models generated. Elapsed time: {(t1 - t0)} seconds.")

    # Extract frame-level state labels applying Viterbi algorithm on each audio and the speaker's acoustic model
    train_prebatch_mfccs = train_prebatch_mfccs.map(generate_state_labels_mfccs, num_parallel_calls=AUTOTUNE)
    train_prebatch_mfccs = train_prebatch_mfccs.shuffle(
        buffer_size=BUFFER_SIZE,
        reshuffle_each_iteration=True
    )

    # TODO: (maybe) one-hot encode the state labels

    # Save output datasets
    print("Extracting state labels and saving results...")
    tf.data.experimental.save(train_prebatch_mfccs, path=TRAIN_SET_PATH_MFCCS_TF)
    # TODO: save the other datasets
    print("Preprocessing completed.")


if __name__ == "__main__":
    main()
