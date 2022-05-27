import os
import numpy as np
import librosa
import librosa.display
from pydub import AudioSegment, silence
import soundfile as sf
from typing import final

TRAIN_PERCENTAGE: final = 0.75
MIN_SILENCE_LEN: final = 500
SILENCE_THRESHOLD: final = -16
SILENCE_SEEK_STEP: final = 2


def remove_silence(path: str, export_path: str = None) -> (np.ndarray, int):
    """
    Loads audio file in given path, removes silence frames longer than MIN_SILENCE_LEN, not louder than
    SILENCE_THRESHOLD dB and with a SILENCE_SEEK_STEP of two frames, returning and exporting the new sound file to an
    export path, if given.

    :param path: path to the audio file to remove silence from.
    :param export_path: export path for the silence-cleaned audio file (default None).
    :return: silence-cleaned audio file and the audio sampling rate.
    """
    # Check if export path exist
    if export_path is not None and not os.path.exists(export_path):
        # Create a new directory if it doesn't exist
        os.makedirs(export_path)

    # Read the audio file
    data, sr = librosa.load(path)

    # Filename extraction from path
    filename = os.path.basename(path)
    # Save temporary file wav with rfidd if export directory is not None
    if export_path is not None:
        sf.write(export_path + filename, data, sr)
        data_as = AudioSegment.from_wav(export_path + filename)
    else:
        data_as = AudioSegment.from_wav(path)

    # Detect silence intervals where silence last 500ms and decibel range reduction is higher than 16dB
    silence_ranges = silence.detect_silence(
        data_as,
        min_silence_len=MIN_SILENCE_LEN,
        silence_thresh=SILENCE_THRESHOLD,
        seek_step=SILENCE_SEEK_STEP
    )
    # Generate indexes of silence intervals
    indexes = []
    for sr in silence_ranges:
        indexes = [*indexes, *range(sr[0], sr[1] + 1)]
    # Delete silence interval
    data = np.delete(data, indexes, axis=0)

    # Save wav file if export_path is not None
    if export_path is not None:
        sf.write(export_path + filename, data, sr)

    return data, sr


def fill_audio_frames(audio_frames: np.ndarray, target_len: int, mode: int = 0) -> np.ndarray:
    """
    Fills given audio frame array either with 0s or repeating the frames circularly.

    :param audio_frames: audio frame array (with each frame either containing raw sampled audio data, MFCCs, LPCCs or
                         any other kind of frame-level audio features) to fill until the target size.
    :param mode: either 0 or 1, if 0 audio_frames will be filled with 0-valued frames, if 1 it will be filled repeating
                 audio frames in a circular way.
    :param target_len: target size of the output array.

    :return: a new audio frame array filled until the target size.
    """
    if mode != 0 and mode != 1:
        raise ValueError("Mode must be either 0 or 1.")

    target_audio = np.copy(audio_frames)
    frame_len = len(target_audio[0])
    dist = target_len - len(target_audio)
    added_frames = 0
    fill_frame = None

    while added_frames < dist:
        if mode == 0:
            fill_frame = np.zeros(shape=(1, frame_len))
        if mode == 1:
            fill_frame = np.reshape(np.array(audio_frames[added_frames % len(audio_frames)]), newshape=(1, frame_len))

        target_audio = np.concatenate((target_audio, fill_frame), axis=0)
        added_frames += 1

    return target_audio
