import dnnhmm
from typing import final
import keras
import numpy as np
from training.training_utils import load_dataset, one_hot_labels_to_integer_labels
from preprocessing.file_utils import generate_or_load_speaker_ordered_dict, load_speakers_acoustic_models
from preprocessing.constants import UNSPLITTED_SET_PATH_MFCCS, TRAIN_SET_PATH_MFCCS, TEST_SET_PATH_MFCCS,\
    N_STATES_MFCCS, AUDIO_PER_SPEAKER
from training.training_utils import sparse_top_k_categorical_speaker_accuracy_mfccs, \
    speaker_n_states_in_top_k_accuracy_mfccs, sparse_categorical_speaker_accuracy_mfccs


_EPOCHS_LOAD_RECCONV: final = 1
_VERSION_LOAD_RECCONV: final = 0.4
_RECCONV_NET_PATH: final = f"fitted_mlp_predictor/mlp_predictor_{_EPOCHS_LOAD_RECCONV}_epochs_v{_VERSION_LOAD_RECCONV}"


def _compute_state_frequencies(labels: np.ndarray, audios_per_speaker: int = AUDIO_PER_SPEAKER) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    states, state_frequencies = np.unique(labels, return_counts=True)
    state_relative_frequencies = state_frequencies / (labels.shape[1]*audios_per_speaker)
    return states, state_frequencies, state_relative_frequencies


def main():
    # Load dataset
    train_mfccs, train_mfccs_labels = load_dataset(TRAIN_SET_PATH_MFCCS)
    test_mfccs, test_mfccs_labels = load_dataset(TEST_SET_PATH_MFCCS)

    mfccs, mfccs_labels = load_dataset(UNSPLITTED_SET_PATH_MFCCS)
    
    mfccs_labels = one_hot_labels_to_integer_labels(mfccs_labels)

    speaker_indexes = generate_or_load_speaker_ordered_dict()

    speaker_acoustic_models = load_speakers_acoustic_models(list(speaker_indexes.keys()))

    states, state_frequencies, state_relative_frequencies = _compute_state_frequencies(mfccs_labels, AUDIO_PER_SPEAKER)

    model = keras.models.load_model(_RECCONV_NET_PATH, custom_objects={
        "sparse_top_k_categorical_speaker_accuracy_mfccs":sparse_top_k_categorical_speaker_accuracy_mfccs,
        "speaker_n_states_in_top_k_accuracy_mfccs":speaker_n_states_in_top_k_accuracy_mfccs,
        "sparse_categorical_speaker_accuracy_mfccs":sparse_categorical_speaker_accuracy_mfccs})

    for speaker in speaker_indexes:
        # Calculate state range
        start_range = speaker_indexes[speaker]*N_STATES_MFCCS
        end_range = start_range + N_STATES_MFCCS
        # Slice speaker state frequencies over the range
        speaker_state_frequencies = state_relative_frequencies[start_range:end_range]
        # Load relative speaker
        acoustic_model = speaker_acoustic_models[speaker]
        # Generate the DNNHMM model
        final_model = dnnhmm.DNNHMM(
            transitions=acoustic_model.transitions_,
            priors=acoustic_model.initial_,
            emission_model=model,
            state_frequencies=speaker_state_frequencies
        )
        # Index for audio range 0:10 first speaker etc...
        audio_range = 0
        #iterate over the audio tensors
        for index in range(audio_range, audio_range+10):
            # qui c'è un problema con le dimensioni
            # expand_dims perchè la rete vuole un tensore del tipo (batch_size, timesteps, coeff)
            most_likely_path, most_likely_path_prob = final_model.viterbi(y=np.expand_dims(mfccs[index], axis=0),
                                                                          state_range=(start_range, end_range))
            # stampo i risultati
            print("Most Likely Path")
            print(most_likely_path)
            print("Most Likely Path probabilities")
            print(most_likely_path_prob)
            print("Label calculated")
            print(state_frequencies[index])

if __name__ == "__main__":
    main()
