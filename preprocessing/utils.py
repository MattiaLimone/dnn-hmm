# Loading the Libraries
import os
import numpy as np
from spafe.utils import preprocessing
import soundfile as sf
from typing import final, Optional
import re

TRAIN_PERCENTAGE: final = 0.70
MINIMUM_SILENCE_LENGTH: final = 500
SILENCE_THRESHOLD: final = -35
SEEK_STEP = 2


def remove_silence(path: str) -> (np.ndarray, int):
    """
    Removes each silence frame longer than MINIMUM_SILENCE_LENGTH and not louder than SILENCE_THRESHOLD by an audio file
    from the given path, with a silence seek equal to SEEK_STEP, and then returns the silence-cleaned audio.

    :param path: a string representing the path to the audio file to remove the silence from.
    :return: the silence-cleaned audio file, alongside with the sample rate.
    """

    # Read the Audiofile
    data, sr = sf.read(path)

    '''    
    # Name extraction from path
    filename = os.path.basename(path)

    # Dirname extraction
    tmp_export_dir = os.path.dirname(path) + "/" + _TMP_DIR_NAME
   
    if export_path is not None:
        # Check if export path exist
        if not os.path.exists(export_path):
            # Create a new directory because it does not exist
            os.makedirs(export_path)

        # Save temporary file wav with rfidd
        sf.write(export_path + filename, data, sr)
        data_as = AudioSegment.from_wav(export_path + filename)

    else:
        # Create temporary directory to contain temporary wav
        os.makedirs(tmp_export_dir)

        # Save temporary file wav with rfidd and read it with pydub
        sf.write(tmp_export_dir + "/" + filename, data, sr)
        data_as = AudioSegment.from_wav(tmp_export_dir + "/" + filename)

    # Detect silence intervals where silence last 500ms and dB range reduction is higher than 16dB
    silence_ranges = silence.detect_silence(
        data_as,
        min_silence_len=MINIMUM_SILENCE_LENGTH,
        silence_thresh=SILENCE_THRESHOLD,
        seek_step=SEEK_STEP
    )
    # Generate indexes of silence interval
    indexes = []
    for silence_range in silence_ranges:
        indexes = [*indexes, *range(silence_range[0], silence_range[1] + 1)]
    # Delete silence interval
    data = np.delete(data, indexes, axis=0)
    # Save wav file
    if export_path is not None:
        sf.write(export_path + filename, data, sr)
    else:
        # Delete temporary directory
        shutil.rmtree(tmp_export_dir)
    '''
    energy, vad, data = preprocessing.remove_silence(sig=data, fs=sr, threshold=SILENCE_THRESHOLD)

    return data, sr


def fill_audio_frames(audio_frames: np.ndarray, target_len: int, mode: int = 0) -> np.ndarray:
    """
    Fills given audio frame array either with 0s or repeating the frames circularly.

    :param audio_frames: numpy array representing audio frame array (with each frame either containing raw sampled audio
                         data, MFCCs, LPCCs or any other kind of frame-level audio features) to fill until the target
                         size.
    :param mode: an integer, either 0 or 1, if 0 audio_frames will be filled with 0-valued frames, if 1 it will be
                 filled repeating audio frames in a circular way.
    :param target_len: an integer representing target size of the output array.

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


def speaker_audio_filenames(path: str, speaker_dir_regex: re.Pattern, audio_file_regex: re.Pattern) -> \
        dict[str, list[str]]:
    """
    This function will iterate recursively over the dataset directory, processing each speaker directory whose name
    whose name matches the given speaker_dir_regex, and storing each audio file that matches the given audio_file_regex
    into a dictionary composed of speaker:path_to_audio_files pairs.

    :param path: A string representing the path to the dataset root folder.
    :param speaker_dir_regex: regex to match speaker directories.
    :param audio_file_regex: regex to match audio files in the speaker directories.
    :return: A dictionary of speaker:path_to_audio_files pairs.
    :rtype: dict[str, list[str]]
   """
    speakers_audios_filenames: dict[str, list[str]] = {}
    _speakers_audios_filenames_rec(
        path=path,
        speakers_audios=speakers_audios_filenames,
        speaker_dir_regex=speaker_dir_regex,
        audio_file_regex=audio_file_regex
    )

    return speakers_audios_filenames


def _speakers_audios_filenames_rec(path: str, speakers_audios: dict[str, list[str]], speaker_dir_regex: re.Pattern,
                                   audio_file_regex: re.Pattern, visited: Optional[set] = None):
    """
    This function will iterate recursively over the dataset directory, processing each speaker directory whose name
    whose name matches the given speaker_dir_regex, and storing each audio file that matches the given audio_file_regex
    into a dictionary composed of speaker:path_to_audio_files pairs.

    :param path: A string representing the path to the dataset root folder.
    :param speakers_audios: an empty dictionary in which all speaker-path_to_audio_file pairs will be stored
    :param speaker_dir_regex: regex to match speaker directories.
    :param audio_file_regex: regex to match audio files in the speaker directories.
    :param visited: A set. It's used to mark a specific path as already visited.
    """
    if visited is None:
        visited = set()
    basename = os.path.basename(path)

    if os.path.isdir(path) and basename not in visited:
        visited.add(path)

        # Base case: leaf in searched files
        if speaker_dir_regex.match(basename):

            if basename not in speakers_audios:
                speakers_audios[basename] = []

            for entry in os.listdir(path):
                if audio_file_regex.match(entry):
                    audio_path = path + "/" + entry
                    speakers_audios[basename].append(audio_path)

        # Recursive call
        else:
            for entry in os.listdir(path):
                newpath = path + "/" + entry
                if os.path.isdir(newpath):
                    _speakers_audios_filenames_rec(
                        path=newpath,
                        speakers_audios=speakers_audios,
                        speaker_dir_regex=speaker_dir_regex,
                        audio_file_regex=audio_file_regex,
                        visited=visited
                    )
