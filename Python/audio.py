import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

import numpy as np
import ffmpeg
import librosa
import matplotlib.pyplot as plt




dirpath = 'Data_super_icbs'
group = '20240312_1629_super_5KHZ83'
filename = '1710326137265-4144e390-caf9-40c5-9424-9cc5f734cbb6-cam-audio-1710326138270'
filename_path =  os.path.join(dirpath, group, filename)

print(filename_path)

y, sr = librosa.load(filename_path, sr = 32000)


# tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
# print(tempo)

# # calculate RMS for loudness/energy
# S, phase = librosa.magphase(librosa.stft(y))
# rms = librosa.feature.rms(S=S)

# print(rms)

# # mel-scale for pitch
# mfccs = librosa.feature.mfcc(y, sr=sr)
# print(mfccs)

