from typing import final
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from sequentia import GMMHMM
from preprocessing.constants import AUDIO_DATAFRAME_KEY, TRAIN_SET_PATH_MFCCS, TRAIN_SET_PATH_MEL_SPEC, \
    AUDIO_PER_SPEAKER, N_STATES_MFCCS
from preprocessing.file_utils import generate_or_load_speaker_ordered_dict, load_speakers_acoustic_models
from training.training_utils import load_dataset
from preprocessing.augmentation.augmentation import mixup, DEFAULT_SCALING_FACTOR
from preprocessing.augmentation.augmentation_constants import MAX_MIXUPS, MIXED_UP_MFCCS_DATASET_PATH, \
    MIXED_UP_MEL_SPEC_DATASET_PATH


_RANDOM_SEED: final = 47


def _generate_speaker_mixed_up_audios(feature_dataframes: list[pd.DataFrame], speaker_indexes: dict[str, int],
                                      max_mixups: int = MAX_MIXUPS, scaling_factor: float = DEFAULT_SCALING_FACTOR) \
        -> list[dict[str, np.ndarray]]:
    """
    Applies mixup transofrmation to generate up to max_mixups additional audios per speaker, combining only audios with.

    :param feature_dataframes: list of dataframe containing numpy tensor containing audio features for each utterance
        and corresponding labels.
    :param speaker_indexes: a dictionary indicating the processing order for the speakers.
    :param max_mixups: maximum number mixups for each speaker; if -1 is given, all possible combinations will be used.
    :param scaling_factor: scaling factor used for the mixup operation.
    :return: a list of dictionaries mapping each speaker identifier into a numpy tensor containing the mixed up audios
        generated using their utterances. The list contains an element for each given feature dataframe.
    """

    mixed_up_audios = [{} for _ in range(0, len(feature_dataframes))]
    np.random.seed(_RANDOM_SEED)  # grants same result in multiple executions

    # If max_mixups is not given, then all the possible mixups within speaker audios will be made
    n_mixups = max_mixups if max_mixups > -1 else AUDIO_PER_SPEAKER * AUDIO_PER_SPEAKER / 2
    mixed_up_audios_shape = []

    # Compute mixup tensor shape for each given feature dataframe
    for i in range(0, len(mixed_up_audios)):
        mixed_up_audios_shape.append((n_mixups,) + feature_dataframes[i][AUDIO_DATAFRAME_KEY].loc[0].shape)

    # For each speaker
    for speaker in speaker_indexes:

        # Generate empty tensor to contain the mixed up audios
        for i in range(0, len(mixed_up_audios)):
            mixed_up_audios[i][speaker] = np.zeros(shape=mixed_up_audios_shape)

        # Get speaker index
        speaker_index = speaker_indexes[speaker]

        generated_mixups = set()
        n_generated_mixups = 0
        # While we didn't generate enough mixups
        while n_generated_mixups < n_mixups:

            # Generate a couple of indexes
            audio_index0 = np.random.randint(
                speaker_index * AUDIO_PER_SPEAKER,
                speaker_index * AUDIO_PER_SPEAKER + AUDIO_PER_SPEAKER
            )
            audio_index1 = np.rando.randint(
                speaker_index * AUDIO_PER_SPEAKER,
                speaker_index * AUDIO_PER_SPEAKER + AUDIO_PER_SPEAKER
            )

            # If the generated couple of indexes wasn't already used and indexes are not the same, generate mixup
            if audio_index0 != audio_index1 and ((audio_index0, audio_index1) not in generated_mixups or
                                                 (audio_index1, audio_index0) not in generated_mixups):

                # For each given audio feature dataframe
                for i in range(0, len(feature_dataframes)):
                    # Generate audio mixup
                    mixed_up_audio = mixup(
                        audio_features0=feature_dataframes[i][AUDIO_DATAFRAME_KEY].loc[audio_index0],
                        audio_features1=feature_dataframes[i][AUDIO_DATAFRAME_KEY].loc[audio_index1],
                        scaling_factor=scaling_factor
                    )

                    # Insert generated mixup into the speaker generated mixup tensor
                    mixed_up_audios[i][speaker][n_generated_mixups, :, :] = mixed_up_audio

                # Increment generated mixups count
                n_generated_mixups += 1

                # Store already generated mixups to avoid repetitions
                generated_mixups.add((audio_index0, audio_index1))

    return mixed_up_audios


def _generate_labels(speaker_mixed_up_audio_features: dict[str, np.ndarray], acoustic_models: dict[str, GMMHMM],
                     speaker_indexes: dict[str, int]) -> dict[str, list[lil_matrix]]:
    """
    Generates frame-level state labels starting from the given mixed-up audio features.

    :param speaker_mixed_up_audio_features: a dictionary mapping each speaker identifier into a tensor containing the
        mixed-up audio features generated with that speaker.
    :param acoustic_models: a dictionary mapping each speaker identifier to its GMM-HMM acoustic model.
    :param speaker_indexes: a dictionary indicating the processing order for the speakers.
    :return: a dictionary mapping each speaker identifier to a list of sparse matrices containing the frame-level state
        labels.
    """
    speaker_mixed_up_audio_labels = {}
    n_speakers = len(speaker_indexes)
    max_frames = None

    # For each speaker
    for speaker in speaker_indexes:

        if max_frames is None:
            max_frames = speaker_mixed_up_audio_features[speaker].shape[1]

        speaker_mixed_up_audio_labels[speaker] = []
        acoustic_model = acoustic_models[speaker]

        # Create max_frames x (n_states*n_speaker) matrix representing the one-hot encoding of the state labels
        speaker_one_hot_encoded_state_labels = np.zeros(shape=(max_frames, N_STATES_MFCCS * n_speakers))

        # For each audio of the speaker
        for audio in speaker_mixed_up_audio_features[speaker]:

            # Get raw state index (labels)
            _, raw_states = acoustic_model.model.decode(audio, algorithm='viterbi')

            # For each state decoded, check if it corresponds to the label of the frame in the dataset, replacing raw
            # state index with global state index for each frame
            for i in range(0, len(raw_states)):
                global_state_label = (N_STATES_MFCCS * speaker_indexes[speaker]) + raw_states[i]
                speaker_one_hot_encoded_state_labels[i, global_state_label] = 1

            # Convert the generated one-hot encoded state labels matrix to sparse lil format and store it in the list
            speaker_one_hot_encoded_state_labels = lil_matrix(speaker_one_hot_encoded_state_labels)
            speaker_mixed_up_audio_labels[speaker].append(speaker_one_hot_encoded_state_labels)

    return speaker_mixed_up_audio_labels


def _generate_output_dataframe(speaker_mixed_up_audio_features: dict[str, np.ndarray],
                               speaker_mixed_up_audio_labels: dict[str, list[lil_matrix]]) -> pd.DataFrame:
    """
    Generates output dataframe containing both audio features and labels.

    :param speaker_mixed_up_audio_features: a dictionary mapping each speaker identifier into a tensor containing the
        mixed-up audio features generated with that speaker.
    :param speaker_mixed_up_audio_labels: a dictionary mapping each speaker identifier to a list of sparse matrices
        containing the frame-level state labels.
    :return: a pandas DataFrame containing, on each row, an audio feature tensor and the corresponding frame-level state
        labels.
    """
    # TODO: implement this
    pass


def main():
    # Load training dataset audios
    train_mfccs = load_dataset(TRAIN_SET_PATH_MFCCS, mode=1)
    train_mel_spec = load_dataset(TRAIN_SET_PATH_MEL_SPEC, mode=1)

    # Load speaker dictionary
    speaker_indexes = generate_or_load_speaker_ordered_dict()

    # Load acoustic models
    acoustic_models = load_speakers_acoustic_models(list(speaker_indexes.keys()))

    # Generate mixed up audios
    speakers_mixed_up_audios_mfccs, speakers_mixed_up_audios_mel_spec = _generate_speaker_mixed_up_audios(
        feature_dataframes=[train_mfccs, train_mel_spec],
        speaker_indexes=speaker_indexes,
        max_mixups=MAX_MIXUPS,
        scaling_factor=DEFAULT_SCALING_FACTOR
    )

    # Generate labels for mixed up audios MFCCs features
    speaker_mixed_up_audio_labels = _generate_labels(speakers_mixed_up_audios_mfccs, acoustic_models, speaker_indexes)

    # Generate output dataframe
    output_dataframe_mfccs = _generate_output_dataframe(
        speakers_mixed_up_audios_mfccs,
        speaker_mixed_up_audio_labels,
    )
    output_dataframe_mel_spec = _generate_output_dataframe(
        speakers_mixed_up_audios_mel_spec,
        speaker_mixed_up_audio_labels
    )

    # Save results to pickle files
    output_dataframe_mfccs.to_pickle(MIXED_UP_MFCCS_DATASET_PATH)
    output_dataframe_mel_spec.to_pickle(MIXED_UP_MEL_SPEC_DATASET_PATH)


if __name__ == "__main__":
    main()
