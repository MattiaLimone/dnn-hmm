import os
from preprocessing.constants import VOXCELEB_PATH, LONGEST_TIMIT_AUDIO_PATH, VOXCELEB_OUTPUT_PATH
from preprocessing.utils import create_chuncks


def read_VOXCELEB(path: str):
    audio_list = []
    for audio_wav in os.listdir(path):
        audio_list.append(os.path.join(path, audio_wav))

    return audio_list


if __name__ == "__main__":
    list_vox = read_VOXCELEB(VOXCELEB_PATH)
    for audio in list_vox:
        create_chuncks(longer_path=audio,
                       shorter_path=LONGEST_TIMIT_AUDIO_PATH,
                       export_path=VOXCELEB_OUTPUT_PATH)
