from typing import final, Optional
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import preprocessing.utils as utl
from preprocessing.features.mel import extract_mfccs, MFCC_NUM_DEFAULT, DERIVATIVE_ORDER_DEFAULT
from gmmhmm import gmm_hmm_grid_search
from preprocessing.constants import DATASET_ORIGINAL_PATH
from preprocessing.file_utils import speaker_audio_filenames, SPEAKER_DIR_REGEX, AUDIO_REGEX
from threading import Thread


_N_STATES_MAX_MFCCS: final = 20
_N_MIX_MAX_MFCCS: final = 15
_N_JOBS: final = 4


def _speakers_audios_mfccs_max_frames(speakers_audios_names: dict) -> (dict, int):
    """
    Computes the maximum frames length among all audios, then it generates a dictionary containing speaker-MFCCs pairs
    (for each audio).

    :param speakers_audios_names: A dictionary of speaker-path_to_audio_files pairs.
    :return: A dictionary containing speaker-MFCCs pairs and the maximum frames length.
    """
    speaker_audios_mfccs = {}
    max_frames_mfcc = 0

    # For each speaker
    for speaker in tqdm(speakers_audios_names, desc="Extracting MFCCs and max frame number: "):

        if speaker not in speaker_audios_mfccs:
            speaker_audios_mfccs[speaker] = []

        # Extract MFCCs and LPCCs and search for max frame number
        for audio_path in tqdm(speakers_audios_names[speaker], desc=f"Extracting from audios of speaker: {speaker}"):
            silence_cleaned_audio, sr = utl.remove_silence(audio_path)

            # MFCCs handling
            mfccs = extract_mfccs(silence_cleaned_audio, sr)
            speaker_audios_mfccs[speaker].append(mfccs)

            # Update max frame num if found higher number of frames
            if len(mfccs) > max_frames_mfcc:
                max_frames_mfcc = len(mfccs)

    return speaker_audios_mfccs, max_frames_mfcc


def _fill_speakers_audios_features(speaker_audio_features: dict, max_frames: int, feature_num: int = 0,
                                   mode: int = 0) -> dict:
    """
     Fills each given audio frame array of the input dictionary either with 0s or repeating the frames circularly.

    :param speaker_audio_features: A dictionary of speaker-MFCCs/LPCCs pairs.
    :param max_frames: An integer. The target length to normalize each audio
    :param feature_num: An integer. Number of features calculated for each frame. If given 0, it will be inferred
                        automatically from the length first frame of the first audio (assuming all the frames are the
                        same length).
    :param mode: An integer, either 0 or 1, if 0 each frame will be filled with 0-valued frames, if 1 it will be
                 filled repeating audio frames in a circular fashion.
    :return: A normalized dictionary of speaker-MFCCs/LPCCs pairs
    """
    speaker_audios_features_filled = {}

    # Generate desc string
    mode_string = ""
    if mode == 0:
        mode_string = "with 0-valued frames padding"
    elif mode == 1:
        mode_string = "repeating frames in a circular fashion"
    desc = f"Filling audio {mode_string} with max_frames: {max_frames}, feature_num: {feature_num}"

    # For each speaker
    for speaker in tqdm(speaker_audio_features, desc=desc):
        # If given feature_num is 0, infer feature number by looking at the first audio frame length
        if feature_num == 0:
            feature_num = len(speaker_audio_features[speaker][0][0])

        # Create empty numpy _AUDIO_PER_SPEAKER x max_frames x feature_num tensor to contain the filled audio features
        speaker_audios_features_filled[speaker] = np.zeros(
            shape=(1, max_frames, feature_num),
            dtype=np.float64
        )

        # For each audio of the speaker
        for speaker_audio in speaker_audio_features[speaker]:
            # Fill with zero-value features or in a circular fashion based on the given mode
            filled_audio = utl.fill_audio_frames(speaker_audio, target_len=max_frames, mode=mode)
            speaker_audios_features_filled[speaker] = np.append(
                speaker_audios_features_filled[speaker],
                np.reshape(filled_audio, newshape=(1, max_frames, feature_num)),
                axis=0
            )

        # Remove first, empty matrix from the result tensor
        speaker_audios_features_filled[speaker] = speaker_audios_features_filled[speaker][1:]

    return speaker_audios_features_filled


def _acoustic_model_grid_search_multithread(speakers_audios_features: dict, n_states_max: int, n_mix_max: int,
                                            n_jobs: int = 1) -> pd.DataFrame:
    """
    Executes grid search to find best parameters for each speaker acoustic model with multithreading.

    :param speakers_audios_features:  A dictionary of speaker-MFCCs/LPCCs/Mel-spectrogram pairs.
    :param n_states_max: number of states to generate the acoustic model.
    :param n_mix_max: number of mixtures for each state.
    :param n_jobs: number of threads to create to faster the grid search (by default, 1 is used).
    :return: a pandas DataFrame containing best n_states/n_mix combination for acoustic model alongside best model
        score.
    :raises ValueError: if n_jobs <= 0.
    """
    if n_jobs <= 0:
        raise ValueError("n_jobs must be strictly positive")

    if n_jobs > 1:
        speakers = list(speakers_audios_features.keys())
        n_speaker_per_job = len(speakers) // n_jobs
        threads = []
        results = [None for _ in range(0, n_jobs)]

        # Create jobs from 1 to n_jobs-1 splitting
        for i in range(0, n_jobs - 1):
            target_speakers = speakers[i * n_speaker_per_job:i * n_speaker_per_job + n_speaker_per_job]
            thread = Thread(
                target=__acoustic_model_grid_search_single_thread,
                args=(speakers_audios_features, n_states_max, n_mix_max, target_speakers, i, results),
                name=f"grid_search_thread{i}"
            )
            threads.append(thread)

        # Create last job
        target_speakers = speakers[(n_jobs - 1) * n_speaker_per_job:]
        last_thread = Thread(
            target=__acoustic_model_grid_search_single_thread,
            args=(speakers_audios_features, n_states_max, n_mix_max, target_speakers),
            name=f"grid_search_thread{n_jobs - 1}"
        )
        threads.append(last_thread)

        # Execute threads
        for i in range(n_jobs):
            threads[i].start()

        # Wait end of execution
        for i in range(n_jobs):
            threads[i].join()

        # Combine results
        final_result = results[0]
        for i in range(1, n_jobs):
            final_result = pd.concat([final_result, results[i]], axis=0)

        return final_result

    else:
        return __acoustic_model_grid_search_single_thread(speakers_audios_features, n_states_max, n_mix_max)


def __acoustic_model_grid_search_single_thread(speakers_audios_features: dict, n_states_max: int, n_mix_max: int,
                                               target_speakers: Optional[list[str]] = None, index: Optional[int] = None,
                                               results: Optional[list[pd.DataFrame]] = None) -> pd.DataFrame:
    """
    Executes grid search to find best parameters for each speaker acoustic model.

    :param speakers_audios_features:  A dictionary of speaker-MFCCs/LPCCs/Mel-spectrogram pairs.
    :param n_states_max: number of states to generate the acoustic model.
    :param n_mix_max: number of mixtures for each state.
    :param target_speakers: a list of speaker identifiers to take into account when executing acoustic model grid
        search (if given None, all speakers shall be taken into account).
    :param results: list of results to store the results (for multithreading purposes).
    :param index: index of the results list to store the result into (for multithreading purposes).
    :return: a pandas DataFrame containing best n_states/n_mix combination for acoustic model alongside best model
        score.
    :raises ValueError: if index is out of bounds of results, or if it is None and results is not.
    """
    if results is not None and index is None:
        raise ValueError("index cannot be None while passing results argument")
    if index is not None:
        if results is not None and (index < 0 or index >= len(results)):
            raise ValueError(f"index must be between {0} and {len(results)} if given")

    if target_speakers is None:
        target_speakers = list(speakers_audios_features.keys())

    desc = f"Executing grid search for speaker-acoustic_models" \
           f" best params with n_states_max: {n_states_max}, n_mix: {n_mix_max}"
    best_params = pd.DataFrame(columns=["n_states", "n_mix", "score"])

    # For each speaker to take into account
    for speaker in tqdm(target_speakers, desc=desc):
        print(f"\nCurrent speaker: {str(speaker)}")
        speaker_audios = speakers_audios_features[speaker]

        # Find the best params and relative score
        _, best_speaker_params, score = gmm_hmm_grid_search(
            X=speaker_audios,
            min_state_number=1,
            max_state_number=10,
            min_mix_number=1,
            max_mix_number=n_mix_max,
            verbose=False
        )

        # Store the found params and score to pandas DataFrame
        best_params.loc[speaker] = [best_speaker_params["n_states"], best_speaker_params["n_mix"], score]

    if results is not None:
        results[index] = best_params

    return best_params


def main():
    # Get audio paths, grouped by speaker
    speakers_audios_names = speaker_audio_filenames(
        path=DATASET_ORIGINAL_PATH,
        speaker_dir_regex=SPEAKER_DIR_REGEX,
        audio_file_regex=AUDIO_REGEX
    )

    # Get max frame len audio for each speaker
    speaker_audios_mfccs, max_frames_mfcc = _speakers_audios_mfccs_max_frames(speakers_audios_names)

    # Fill speaker audios in a circular fashion up to the max frame number
    speaker_audios_mfcc_filled_circular = _fill_speakers_audios_features(
        speaker_audios_mfccs,
        max_frames_mfcc,
        feature_num=MFCC_NUM_DEFAULT * (DERIVATIVE_ORDER_DEFAULT + 1),
        mode=1
    )

    # Grid search for acoustic models
    best_params = _acoustic_model_grid_search_multithread(
        speaker_audios_mfcc_filled_circular,
        n_states_max=_N_STATES_MAX_MFCCS,
        n_mix_max=_N_MIX_MAX_MFCCS,
        n_jobs=_N_JOBS
    )

    print("n_mix quantiles: ")
    print(np.quantile(best_params["n_mix"].to_numpy(), [0, 0.25, 0.5, 0.75, 1]))

    print("n_states quantiles: ")
    print(np.quantile(best_params["n_states"].to_numpy(), [0, 0.25, 0.5, 0.75, 1]))

    print("score quantiles: ")
    print(np.quantile(best_params["score"].to_numpy(), [0, 0.25, 0.5, 0.75, 1]))


if __name__ == "__main__":
    main()
