# Loading the Libraries
import os
import numpy as np
import librosa
import librosa.display
from pydub import AudioSegment, silence
import soundfile as sf
from typing import final

TRAIN_PERCENTAGE: final = 0.75
MINIMUM_SILENCE_LENGTH: final = 500
SILENCE_THRESHOLD: final = -16
SEEK_STEP = 2

def remove_silence(path: str, export_path: str = 'export/', do_export: bool = False, min_silence_len: int = MINIMUM_SILENCE_LENGTH,
                   silence_thresh: int = SILENCE_THRESHOLD, seek_step: int = SEEK_STEP) -> (np.ndarray, int):
    """
    Function to detect and remove the silence intervals. Silence is detected as the sound lower than a given threshold
    and that last a minimum time given as function input.

    :param path: A string. The path to the audio file
    :param export_path: A string. The path where to store the modified audio
    :param do_export: A boolean. If True the function save the modified audio to export_path
    :param min_silence_len: An integer. The minimum length for any silent section
    :param silence_thresh: An integer. The upper bound for how quiet is silent in dFBS
    :param seek_step: An integer. Step size for interating over the segment in ms
    :return: The audio without detected silence intervals and sample rate
    """

    # Check if export path exist
    if not os.path.exists(export_path):
        # Create a new directory because it does not exist
        os.makedirs(export_path)
    # Read the Audiofile
    data, samplerate = librosa.load(path)
    # Name extraction from path
    filename = os.path.basename(path)
    # Save temporary file wav with rfidd
    sf.write(export_path + filename, data, samplerate)
    data_as = AudioSegment.from_wav(export_path + filename)
    # Detect silence intervals where silence last 500ms and decibel range reduction is higher than 16dB
    silence_ranges = silence.detect_silence(data_as, min_silence_len=min_silence_len,
                                            silence_thresh=silence_thresh, seek_step=seek_step)
    # Generate indexes of silence interval
    indexes = []
    for sr in silence_ranges:
        indexes = [*indexes, *range(sr[0], sr[1] + 1)]
    # Delete silence interval
    data = np.delete(data, indexes, axis=0)
    # Save wav file
    if(do_export):
        sf.write(export_path + filename, data, samplerate)

    return data, samplerate


def fill_audio_frames(audio_frames: np.ndarray, target_len: int, mode: int = 0) -> np.ndarray:
    """
    Fills given audio frame array either with 0s or repeating the frames circularly.

    :param audio_frames: A Numpy Array. The audio frame array (with each frame either containing raw sampled audio data,
                        MFCCs, LPCCs or any other kind of frame-level audio features) to fill until the target size.
    :param mode: An interger. Either 0 or 1, if 0 audio_frames will be filled with 0-valued frames, if 1 it will be filled repeating
                 audio frames in a circular way.
    :param target_len: An integer. The target size of the output array.

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
