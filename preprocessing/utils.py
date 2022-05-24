# Loading the Libraries
import os
import numpy as np
import librosa
import librosa.display
from pydub import AudioSegment, silence
import soundfile as sf
from typing import final

MEL_COEF_NUM_DEFAULT: final = 13
TRAIN_PERCENTAGE: final = 0.75


# Function to detect and remove the silence intervals where silence
# last 500ms and decibel range reduction is higher than 16dB
def remove_silence(path: str, export_path: str = 'export/'):
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
    silence_ranges = silence.detect_silence(data_as, min_silence_len=500, silence_thresh=-16, seek_step=2)
    # Generate indexes of silence interval
    indexes = []
    for sr in silence_ranges:
        indexes = [*indexes, *range(sr[0], sr[1] + 1)]
    # Delete silence interval
    data = np.delete(data, indexes, axis=0)
    # Save wav file
    sf.write(export_path + filename, data, samplerate)
    return data, samplerate




