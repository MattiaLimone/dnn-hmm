from typing import final
import keras
import numpy as np
import dnnhmm
from training.training_utils import TRAIN_SET_PATH_MFCCS, TEST_SET_PATH_MFCCS, TRAIN_SET_PATH_MEL_SPEC, \
    TEST_SET_PATH_MEL_SPEC, load_dataset, one_hot_labels_to_integer_labels
from preprocessing.file_utils import generate_or_load_speaker_ordered_dict, load_speakers_acoustic_models
from preprocessing.acoustic_model.gmmhmm import load_acoustic_model
from preprocessing.constants import STATE_PROB_KEY, AUDIO_DATAFRAME_KEY, N_STATES_MFCCS, AUDIO_PER_SPEAKER

_EPOCHS_LOAD_RECCONV: final = 1
_VERSION_LOAD_RECCONV: final = 0.4
_RECCONV_NET_PATH: final = f"fitted_mlp_predictor/mlp_predictor_{_EPOCHS_LOAD_RECCONV}_epochs_v{_VERSION_LOAD_RECCONV}"

def _compute_state_frequencies(labels: np.ndarray, audios_per_speaker: int = AUDIO_PER_SPEAKER) -> tuple[np.ndarray, np.ndarray]:
    states, state_frequencies = np.unique(labels, return_counts=True)
    state_frequencies = state_frequencies.astype(dtype=np.float64)/float(labels.shape[1]*audios_per_speaker)
    return states, state_frequencies

def main():
    # Load dataset
    train_mfccs, train_mfccs_labels = load_dataset(TRAIN_SET_PATH_MFCCS)
    test_mfccs, test_mfccs_labels = load_dataset(TEST_SET_PATH_MFCCS)
    mfccs_labels = one_hot_labels_to_integer_labels(np.concatenate((train_mfccs_labels, test_mfccs_labels), axis=0))

    speaker_indexes = generate_or_load_speaker_ordered_dict()

    speaker_acoustic_models = load_speakers_acoustic_models(list(speaker_indexes.keys()))

    states, state_frequencies = _compute_state_frequencies(mfccs_labels, AUDIO_PER_SPEAKER)

    model = keras.models.load_model(_RECCONV_NET_PATH)
    for speaker in speaker_indexes:
        start_range = speaker_indexes[speaker]*N_STATES_MFCCS
        end_range = start_range + N_STATES_MFCCS
        speaker_state_frequencies = state_frequencies[start_range:end_range]
        acoustic_model = speaker_acoustic_models[speaker]
        final_model = dnnhmm.DNNHMM(
            transitions=acoustic_model.transitions_,
            priors=acoustic_model.initial_,
            emission_model=model,
            state_frequencies=speaker_state_frequencies
        )

if __name__ == "__main__":
    main()
