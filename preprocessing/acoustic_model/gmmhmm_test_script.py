import os
import sys
from glob import glob
import numpy as np
from tqdm.auto import tqdm
import preprocessing.utils as utl
from preprocessing.features.mel import extract_mfccs, extract_mel_spectrum, MEL_FILTER_BANK_DEFAULT, MFCC_NUM_DEFAULT
from preprocessing.features.lpcc import extract_lpccs, LPCC_NUM_DEFAULT
from gmmhmm import generate_acoustic_model, gmm_hmm_grid_search


def main():
    np.set_printoptions(threshold=sys.maxsize)
    # Convert
    results = [y for x in os.walk('data/lisa/data/timit/raw/TIMIT/TEST/DR1/FAKS0') for y in
               glob(os.path.join(x[0], '*.WAV'))]
    print(os.getcwd())
    print(len(results))
    mfccs = {}
    lpccs = {}
    mel_spectrums = {}

    train_set_array_mfcc = np.zeros((1, MFCC_NUM_DEFAULT*3), dtype=np.longfloat)
    train_set_lengths_mfcc = np.zeros(1, dtype=int)
    train_set_array_lpcc = np.zeros((1, LPCC_NUM_DEFAULT), dtype=np.longfloat)
    train_set_lengths_lpcc = np.zeros(1, dtype=int)
    train_set_array_mel_spec = np.zeros((1, MEL_FILTER_BANK_DEFAULT), dtype=np.longfloat)
    train_set_lengths_mel_spec = np.zeros(1, dtype=int)

    for path in tqdm(results):
        filename = str(os.path.basename(path))
        data, sr = utl.remove_silence(path=path, export_path='data/test/')

        # MFCCs
        mfcc = extract_mfccs(signal=data, sr=sr, n_mfcc=MFCC_NUM_DEFAULT)
        train_set_lengths_mfcc = np.append(train_set_lengths_mfcc, len(mfcc))
        train_set_array_mfcc = np.concatenate((train_set_array_mfcc, mfcc), axis=0)
        mfccs[filename] = mfcc

        # LPCCs
        lpcc = extract_lpccs(signal=data, sr=sr, n_lpcc=LPCC_NUM_DEFAULT)
        train_set_lengths_lpcc = np.append(train_set_lengths_lpcc, len(lpcc))
        train_set_array_lpcc = np.concatenate((train_set_array_lpcc, lpcc), axis=0)
        lpccs[filename] = lpcc

        # Mel-mel_spectrum
        mel_spectrum = extract_mel_spectrum(signal=data, sr=sr, n_filter_bank=MEL_FILTER_BANK_DEFAULT)
        train_set_lengths_mel_spec = np.append(train_set_lengths_mel_spec, len(mel_spectrum))
        train_set_array_mel_spec = np.concatenate((train_set_array_mel_spec, mel_spectrum), axis=0)
        mel_spectrums[filename] = mel_spectrum

    train_set_array_mfcc = train_set_array_mfcc[1:, ]
    train_set_lengths_mfcc = train_set_lengths_mfcc[1:]
    train_set_array_lpcc = train_set_array_lpcc[1:, ]
    train_set_lengths_lpcc = train_set_lengths_lpcc[1:]
    train_set_array_mel_spec = train_set_array_mel_spec[1:, ]
    train_set_lengths_mel_spec = train_set_lengths_mel_spec[1:]

    print("Sequence Length Shape: " + str(train_set_lengths_mfcc.shape))
    print("Shape: " + str(train_set_array_mfcc.shape))

    # MFCCs acoustic models
    gmmhmm_model, all_audios_states = generate_acoustic_model(train_set_array_mfcc, train_set_lengths_mfcc,
                                                              n_components=5, n_mix=7)
    print("MFCCs: ")
    print(train_set_lengths_mfcc)
    print([len(audio_states) for audio_states in all_audios_states])
    print(gmmhmm_model.score(train_set_array_mfcc, train_set_lengths_mfcc))

    grid_search_type = int(input("Insert grid search type (-1 no grid search, 0 MFCCs, 1 LPCCs, 2 Mel-Spectrum): "))

    if grid_search_type == 0:
        # TODO: write automated grid search for dataset sample
        best_gmmhmm_model, best_score, best_params = gmm_hmm_grid_search(train_set_array_mfcc, train_set_lengths_mfcc,
                                                                         min_state_number=1, max_state_number=7,
                                                                         min_mix_number=1, max_mix_number=8,
                                                                         min_iter_number=10, max_iter_number=20,
                                                                         verbose=True)
        print("MFCCs best scores and params: ")
        print(best_score)
        print(best_params)

    # Mel-spectrum acoustic models
    gmmhmm_model, all_audios_states = generate_acoustic_model(train_set_array_mel_spec, train_set_lengths_mel_spec,
                                                              n_components=5, n_mix=2)
    print("Mel-spectrum: ")
    print(train_set_lengths_mel_spec)
    print([len(audio_states) for audio_states in all_audios_states])
    print(gmmhmm_model.score(train_set_array_mel_spec, train_set_lengths_mel_spec))

    if grid_search_type == 1:
        # TODO: write automated grid search for dataset sample
        best_gmmhmm_model, best_score, best_params = gmm_hmm_grid_search(
            train_set_array_mel_spec,
            train_set_lengths_mel_spec,
            min_state_number=2,
            max_state_number=4,
            min_mix_number=1, max_mix_number=2,
            min_iter_number=5,
            max_iter_number=10,
            verbose=True
        )

        print("Mel-spectrum best scores and params: ")
        print(best_score)
        print(best_params)

    # LPCCs acoustic models
    gmmhmm_model, all_audios_states = generate_acoustic_model(train_set_array_lpcc, train_set_lengths_lpcc,
                                                              n_components=4, n_mix=3)
    print("LPCCs: ")
    print(train_set_lengths_mel_spec)
    print([len(audio_states) for audio_states in all_audios_states])
    print(gmmhmm_model.score(train_set_array_lpcc, train_set_lengths_lpcc))

    if grid_search_type == 2:
        # TODO: write automated grid search for dataset sample
        best_gmmhmm_model, best_score, best_params = gmm_hmm_grid_search(
            train_set_array_lpcc,
            train_set_lengths_lpcc,
            min_state_number=2,
            max_state_number=4,
            min_mix_number=1, max_mix_number=3,
            min_iter_number=5,
            max_iter_number=10,
            verbose=True
        )

        print("Mel-spectrum best scores and params: ")
        print(best_score)
        print(best_params)


if __name__ == "__main__":
    main()
