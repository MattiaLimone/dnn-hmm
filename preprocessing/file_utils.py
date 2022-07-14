import os
import pickle
import re
from collections import OrderedDict
from typing import Optional, final
from sequentia import GMMHMM
from preprocessing.acoustic_model.gmmhmm import load_acoustic_model
from preprocessing.constants import ACOUSTIC_MODEL_PATH_MFCCS


SPEAKER_DIR_REGEX: final = re.compile("[A-Z]{4}[0-9]")
AUDIO_REGEX: final = re.compile("(.*)\\.WAV")
_SPEAKER_INDEXES_PATH: final = "data/speakerindexes/speakerindexes.pkl"


def speaker_audio_filenames(path: str, speaker_dir_regex: re.Pattern, audio_file_regex: re.Pattern) -> \
        dict[str, list[str]]:
    """
    This function will iterate recursively over the dataset directory, processing each speaker directory whose name
    matches the given speaker_dir_regex, and storing each audio file that matches the given audio_file_regex
    into a dictionary composed of speaker:path_to_audio_files pairs.

    :param path: A string representing the path to the dataset root folder.
    :param speaker_dir_regex: regex to match speaker directories.
    :param audio_file_regex: regex to match audio files in the speaker directories.
    :return: A dictionary of speaker:path_to_audio_files pairs.
    :rtype: dict[str, list[str]]
   """
    speakers_audios_filenames: dict[str, list[str]] = {}
    __speakers_audios_filenames_rec(
        path=path,
        speakers_audios=speakers_audios_filenames,
        speaker_dir_regex=speaker_dir_regex,
        audio_file_regex=audio_file_regex
    )

    return speakers_audios_filenames


def __speakers_audios_filenames_rec(path: str, speakers_audios: dict[str, list[str]], speaker_dir_regex: re.Pattern,
                                    audio_file_regex: re.Pattern, visited: Optional[set] = None):
    """
    This function will iterate recursively over the dataset directory, processing each speaker directory whose name
    matches the given speaker_dir_regex, and storing each audio file that matches the given audio_file_regex into a
    dictionary composed of speaker:path_to_audio_files pairs.

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
                    __speakers_audios_filenames_rec(
                        path=newpath,
                        speakers_audios=speakers_audios,
                        speaker_dir_regex=speaker_dir_regex,
                        audio_file_regex=audio_file_regex,
                        visited=visited
                    )


def generate_or_load_speaker_ordered_dict(speakers: Optional[list] = None, generate: bool = False) -> OrderedDict:
    """
    Generates or loads from file an ordered dictionary of speaker:index pairs.

    :param speakers: A list of keys that represent each speaker.
    :param generate: A boolean. If True it generates the dictionary, otherwise it loads the dictionary from file.
    :return: An ordered dictionary speaker -> index (generating and saving it if it doesn't exist or generate flag is
        given.
    :raises ValueError: if speakers is given None and speaker indexes dictionary has to be generated.
    """
    speaker_indexes = OrderedDict()
    generate_flag = generate

    # If generate flag is False, try to load speaker indexes file, otherwise set generate flag to True
    if not generate:
        try:
            with open(_SPEAKER_INDEXES_PATH, "rb") as file:
                speaker_indexes = pickle.load(file)
        except IOError:
            generate_flag = True

    # If generate flag is True, generate new speaker indexes OrderedDict and save it to file with pickle
    if generate_flag:
        if speakers is None:
            raise ValueError("Cannot generate speaker indexes dictionary if speaker names are not given")
        for i in range(0, len(speakers)):
            speaker_indexes[speakers[i]] = i

        with open(_SPEAKER_INDEXES_PATH, "wb") as file:
            pickle.dump(speaker_indexes, file, protocol=pickle.HIGHEST_PROTOCOL)

    return speaker_indexes


def load_speakers_acoustic_models(speakers: list[str]) -> dict[str, GMMHMM]:
    """
    Loads all the speakers GMM-HMM acoustic models.

    :param speakers: a list containing string speaker identifiers.
    :return: a dictionary mapping each speaker identifier to a GMMHMM acoustic model.
    """
    acoustic_models = {}

    # For each speaker, load MFCCs acoustic model
    for speaker in speakers:
        path = f"{ACOUSTIC_MODEL_PATH_MFCCS}{speaker}.pkl"
        acoustic_models[speaker] = load_acoustic_model(path)

    return acoustic_models
