# Loading the Libraries
import os
import numpy as np
import librosa
from pydub import AudioSegment, silence
import soundfile as sf

def remove_silence(path, export_path = 'export/'):
    # Check if export path exist
    if not os.path.exists(export_path):
        # Create a new directory because it does not exist
        os.makedirs(export_path)
    # Read the Audiofile
    data, samplerate = librosa.load(path)
    # Name extraction from path
    filename = path.split('/')
    filename = filename[len(filename)-1]
    # Save temporary file wav with rfidd
    sf.write(export_path + filename, data, samplerate)
    data_as = AudioSegment.from_wav(export_path + filename)
    # Detect silence intervals where silence last 500ms and decibel range reduction is higher than 16dB
    silence_ranges = silence.detect_silence(data_as, min_silence_len=500, silence_thresh=-16, seek_step=2)
    # Generate indexes of silence interval
    indexes = []
    for sr in silence_ranges:
        indexes = [*indexes,*range(sr[0],sr[1]+1)]
    # Delete silence interval
    data = np.delete(data, indexes, axis=0)
    # Save wav file
    sf.write(export_path + filename, data, samplerate)

