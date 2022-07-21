from models.dnnhmm.dnnhmm import DNNHMM
from models.recconvsnet.recconvsnet import load_recconvsnet
from preprocessing.utils import compute_state_frequencies
from typing import final
from tqdm.auto import tqdm
from training.training_utils import load_dataset, one_hot_labels_to_integer_labels
from preprocessing.file_utils import generate_or_load_speaker_ordered_dict, load_speakers_acoustic_models, \
    load_state_frequencies, save_state_frequencies
from preprocessing.constants import UNSPLITTED_SET_PATH_MFCCS, N_STATES_MFCCS, AUDIO_PER_SPEAKER, TEST_SET_PATH_MFCCS, \
    STATE_FREQUENCIES_PATH, TEST_SET_PATH_MEL_SPEC
from training.training_utils import sparse_top_k_categorical_speaker_accuracy_mfccs, \
    speaker_n_states_in_top_k_accuracy_mfccs
import re


_EPOCHS_LOAD_RECCONV: final = 600
_VERSION_LOAD_RECCONV: final = 1.1
_RECCONV_NET_PATH: final = f"fitted_recconvsnet/recconvsnet_{_EPOCHS_LOAD_RECCONV}_epochs_v{_VERSION_LOAD_RECCONV}"
_VERBOSE: final = False
_AUDIO_START_INDEX: final = 0
_COUNT_START: final = 0
_FEMALE_COUNT_START: final = 0
_MALE_COUNT_START: final = 0
_FEMALE_COUNT_START_TOTAL: final = 0
_MALE_COUNT_START_TOTAL: final = 0
_FEMALE_REGEX: final = re.compile("F[A-Z]{3}[0-9]")
_MALE_REGEX: final = re.compile("M[A-Z]{3}[0-9]")


def main():
    # Load test dataset
    test_mfccs, test_mfccs_labels = load_dataset(TEST_SET_PATH_MFCCS)
    test_mel_spec, _ = load_dataset(TEST_SET_PATH_MEL_SPEC)

    # Convert one-hot encoded labels to integer labels for test set
    test_mfccs_labels = one_hot_labels_to_integer_labels(test_mfccs_labels)

    # Load speaker indexes
    speaker_indexes = generate_or_load_speaker_ordered_dict()
    speaker_keys = list(speaker_indexes.keys())

    # Load acoustic models
    speaker_acoustic_models = load_speakers_acoustic_models(list(speaker_indexes.keys()))

    # Generate or load state frequencies
    states, state_frequencies, state_relative_frequencies = None, None, None
    state_frequencies_tuple = load_state_frequencies(STATE_FREQUENCIES_PATH)
    validation_limit = int(len(test_mfccs)/2)

    # If state frequencies were not generated yet
    if state_frequencies_tuple is None:

        # Load unsplitted dataset
        _, mfccs_labels = load_dataset(UNSPLITTED_SET_PATH_MFCCS)

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

    # Load neural network emission model
    model = load_recconvsnet(path=_RECCONV_NET_PATH, custom_objects={
        "speaker_n_states_in_top_k_accuracy_mfccs": speaker_n_states_in_top_k_accuracy_mfccs,
        "sparse_top_k_categorical_speaker_accuracy_mfccs": sparse_top_k_categorical_speaker_accuracy_mfccs
    })

    speaker_dnnhmms = {}
    # Generate speaker DNNHMM models
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

        # Store the generated model in dictionary
        speaker_dnnhmms[speaker] = final_model

    count = _COUNT_START  # counter for speaker identification match
    male_count_total = _MALE_COUNT_START_TOTAL
    female_count_total = _FEMALE_COUNT_START_TOTAL
    female_count = _FEMALE_COUNT_START
    male_count = _MALE_COUNT_START
    # For each test set audio tensor
    for i in range(validation_limit + _AUDIO_START_INDEX, test_mfccs.shape[0]):
        audio = test_mfccs[i]
        labels = test_mfccs_labels[i]
        best_log_likelihood = None
        best_speaker_match = None

        # Get real speaker index
        speaker_index = int((labels[0] - labels[0] % N_STATES_MFCCS)/N_STATES_MFCCS)
        real_speaker = speaker_keys[speaker_index]
        real_speaker_log_likelihood = None

        # For each speaker
        for speaker in tqdm(speaker_dnnhmms, desc=f"Evaluating performance for {real_speaker}: "):
            # Calculate state range
            start_range = speaker_indexes[speaker]*N_STATES_MFCCS
            end_range = start_range + N_STATES_MFCCS

            # Get speaker DNNHMM model and compute the most likely path with viterbi
            dnnhmm = speaker_dnnhmms[speaker]
            most_likely_path, most_likely_path_prob = dnnhmm.viterbi(
                audio,
                (start_range, end_range),
                'log',
                test_mel_spec[i:i+1]
            )

            if speaker == real_speaker:
                real_speaker_log_likelihood = most_likely_path_prob

            # Update best_log_likelihood if found most_likely_path_prob is greater than current
            if best_log_likelihood is None or most_likely_path_prob > best_log_likelihood:
                best_log_likelihood = most_likely_path_prob
                best_speaker_match = speaker

            if _VERBOSE is True:
                # Print results with log mode (recommended)
                print("Most Likely Path")
                print(most_likely_path)
                print("Most Likely Path probabilities")
                print(most_likely_path_prob)

        print(f"Real speaker: {real_speaker}, "
              f" real speaker log-likelihood: {real_speaker_log_likelihood}, "
              f"best speaker match: {best_speaker_match}, "
              f"log-likelihood: {best_log_likelihood}")

        # Female count
        if _FEMALE_REGEX.match(real_speaker):
            if real_speaker == best_speaker_match:
                female_count += 1
            female_count_total += 1

        # Male count
        elif _MALE_REGEX.match(real_speaker):
            if real_speaker == best_speaker_match:
                male_count += 1
            male_count_total += 1

        # Total count
        if real_speaker == best_speaker_match:
            count += 1

        # break

    accuracy = count / test_mfccs.shape[0]
    male_accuracy = male_count / male_count_total
    female_accuracy = female_count / female_count_total

    print(f"Number of matches: {count}")
    print(f"Accuracy: {accuracy}")
    print(f"Number of male matches: {male_count}")
    print(f"Accuracy: {male_accuracy}")
    print(f"Number of female matches: {female_count}")
    print(f"Accuracy: {female_accuracy}")


if __name__ == "__main__":
    main()
