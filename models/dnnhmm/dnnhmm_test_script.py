from dnnhmm import DNNHMM
from preprocessing.utils import compute_state_frequencies
from typing import final
import keras
from training.training_utils import load_dataset, one_hot_labels_to_integer_labels
from preprocessing.file_utils import generate_or_load_speaker_ordered_dict, load_speakers_acoustic_models, \
    save_state_frequencies, load_state_frequencies
from preprocessing.constants import UNSPLITTED_SET_PATH_MFCCS, N_STATES_MFCCS, AUDIO_PER_SPEAKER, STATE_FREQUENCIES_PATH
from training.training_utils import sparse_top_k_categorical_speaker_accuracy_mfccs, \
    speaker_n_states_in_top_k_accuracy_mfccs, sparse_categorical_speaker_accuracy_mfccs


_EPOCHS_LOAD_RECCONV: final = 1
_VERSION_LOAD_RECCONV: final = 0.4
_RECCONV_NET_PATH: final = f"fitted_mlp_predictor/mlp_predictor_{_EPOCHS_LOAD_RECCONV}_epochs_v{_VERSION_LOAD_RECCONV}"


def main():
    # Load dataset
    mfccs, mfccs_labels = load_dataset(UNSPLITTED_SET_PATH_MFCCS)

    speaker_indexes = generate_or_load_speaker_ordered_dict()

    speaker_acoustic_models = load_speakers_acoustic_models(list(speaker_indexes.keys()))

    # Generate or load state frequencies
    states, state_frequencies, state_relative_frequencies = None, None, None
    state_frequencies_tuple = load_state_frequencies(STATE_FREQUENCIES_PATH)

    # If state frequencies were not generated yet
    if state_frequencies_tuple is None:

        # Convert one-hot encoded labels to integers
        mfccs_labels = one_hot_labels_to_integer_labels(mfccs_labels)

        # Compute state frequencies and store them
        states, state_frequencies, state_relative_frequencies = compute_state_frequencies(
            labels=mfccs_labels,
            audios_per_speaker=AUDIO_PER_SPEAKER
        )
        save_state_frequencies((states, state_frequencies, state_relative_frequencies), STATE_FREQUENCIES_PATH)

    # Otherwise load the existing ones
    else:
        states, state_frequencies, state_relative_frequencies = state_frequencies_tuple

    model = keras.models.load_model(_RECCONV_NET_PATH, custom_objects={
        "sparse_top_k_categorical_speaker_accuracy_mfccs": sparse_top_k_categorical_speaker_accuracy_mfccs,
        "speaker_n_states_in_top_k_accuracy_mfccs": speaker_n_states_in_top_k_accuracy_mfccs,
        "sparse_categorical_speaker_accuracy_mfccs": sparse_categorical_speaker_accuracy_mfccs})

    # Index for audio range 0:10 first speaker etc...
    audio_range = 0
    for speaker in speaker_indexes:
        # Calculate state range
        start_range = speaker_indexes[speaker]*N_STATES_MFCCS
        end_range = start_range + N_STATES_MFCCS

        # Slice speaker state frequencies over the range
        speaker_state_frequencies = state_relative_frequencies[start_range:end_range]

        # Load relative speaker
        acoustic_model = speaker_acoustic_models[speaker]

        # Generate the DNNHMM model
        final_model = DNNHMM(
            transitions=acoustic_model.transitions_,
            priors=acoustic_model.initial_,
            emission_model=model,
            state_frequencies=speaker_state_frequencies
        )

        # Iterate over the audio tensors
        for index in range(audio_range, audio_range + AUDIO_PER_SPEAKER):
            most_likely_path, most_likely_path_prob = final_model.viterbi(
                y=mfccs[index],
                state_range=(start_range, end_range),
                mode='log'
            )
            # Print results
            print("Most Likely Path")
            print(most_likely_path)
            print("Most Likely Path probabilities")
            print(most_likely_path_prob)

            most_likely_path, most_likely_path_prob = final_model.viterbi(
                y=mfccs[index],
                state_range=(start_range, end_range),
                mode='mult'
            )
            # Print results
            print("Most Likely Path")
            print(most_likely_path)
            print("Most Likely Path probabilities")
            print(most_likely_path_prob)
            break

        audio_range += AUDIO_PER_SPEAKER


if __name__ == "__main__":
    main()
