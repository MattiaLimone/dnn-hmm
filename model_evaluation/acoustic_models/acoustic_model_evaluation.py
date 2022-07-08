from sequentia import GMMHMM
from preprocessing.acoustic_model.gmmhmm import load_acoustic_model
from typing import final
from tqdm.auto import tqdm
import pandas as pd
from preprocessing.constants import ACOUSTIC_MODEL_PATH, TRAIN_SET_PATH_MFCCS, \
    TEST_SET_PATH_MFCCS, AUDIO_PER_SPEAKER, AUDIO_DATAFRAME_KEY, STATE_PROB_KEY, N_STATES_MFCCS
from preprocessing.file_utils import generate_or_load_speaker_ordered_dict
from training.training_utils import load_dataset, one_hot_labels_to_integer_labels


_RANDOM_SEED: final = 47
_N_STATES_MAX_MFCCS: final = 20
_N_MIX_MAX_MFCCS: final = 15


def _load_speakers_acoustic_models(speakers: list[str]) -> dict[str, GMMHMM]:
    """
    Loads all the speakers GMM-HMM acoustic models.

    :param speakers: a list containing string speaker identifiers.
    :return: a dictionary mapping each speaker identifier to a GMMHMM acoustic model.
    """
    acoustic_models = {}

    # For each speaker
    for speaker in speakers:
        path = f"{ACOUSTIC_MODEL_PATH}{speaker}.pkl"
        acoustic_models[speaker] = load_acoustic_model(path)

    return acoustic_models


def _check_labels(speaker_indexes: dict[str, int], acoustic_models: dict[str, GMMHMM], dataset: pd.DataFrame):

    """
    Checks equality between frame-level state labels stored in a dataframe and the ones generated by the given GMM-HMMM
    acoustic models through viterbi algorithm.

    :param speaker_indexes: a dictionary mapping each speaker identifier to an unique index.
    :param acoustic_models: a dictionary mapping each speaker in the respective GMM-HMM acoustic model.
    :param dataset: pandas DataFrame containing audio features and respective labels.
    """

    # For each speaker
    for speaker in tqdm(speaker_indexes, desc="Checking validity of labels"):

        # Get speaker audio range
        speaker_index = speaker_indexes[speaker]
        acoustic_model = acoustic_models[speaker]

        # For each audio of that speaker
        for audio_index in range(speaker_index*AUDIO_PER_SPEAKER, speaker_index*AUDIO_PER_SPEAKER + AUDIO_PER_SPEAKER):

            # Extract audio features and
            audio = dataset[AUDIO_DATAFRAME_KEY][audio_index]
            labels = one_hot_labels_to_integer_labels([dataset[STATE_PROB_KEY][audio_index]])[0]
            _, raw_states = acoustic_model.model.decode(audio, algorithm='viterbi')

            # For each state decoded, check if it corresponds to the label of the frame in the dataset, replacing raw
            # state index with global state index
            for i in range(0, len(raw_states)):
                global_state_label = (N_STATES_MFCCS * speaker_indexes[speaker]) + raw_states[i]
                print(f"Dataframe content: {labels[i]}")
                print(f"Acoustic model prediction: {global_state_label}")
                if global_state_label != labels[i]:
                    print(f"Warning: dataframe label '{labels[i]}' != acoustic model prediction '{global_state_label}'")


def main():

    # Get speaker names through OrderedDict
    speaker_indexes = generate_or_load_speaker_ordered_dict()

    # Load dataset
    train_mfccs = load_dataset(TRAIN_SET_PATH_MFCCS, mode=1)
    test_mfccs = load_dataset(TEST_SET_PATH_MFCCS, mode=1)

    full_dataset_mfccs = pd.concat([train_mfccs, test_mfccs], axis=0)

    # Load acoustic models
    acoustic_models = _load_speakers_acoustic_models(list(speaker_indexes.keys()))

    # Check validity of saved labels
    _check_labels(speaker_indexes, acoustic_models, full_dataset_mfccs)


if __name__ == "__main__":
    main()
