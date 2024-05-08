import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
# os.environ["IMAGEIO_FFMPEG_EXE"] = "./.venv/lib/python3.10/site-packages/ffmpeg"

import numpy as np
import ffmpeg
import librosa
import matplotlib.pyplot as plt
import scipy.io as sio

dirpath = 'Data_super_icbs'
group = '20240312_1629_super_5KHZ83'
filename = '1710326137265-4144e390-caf9-40c5-9424-9cc5f734cbb6-cam-audio-1710326138270.wav'
# filename = '1710326137265-a2bbfe30-417d-4fec-84cc-823ac27e1ec3-cam-audio-1710326138271'
filename_path =  os.path.join(dirpath, group, filename)

print(filename_path)

y, sr = librosa.load(filename_path)

tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print(tempo, beat_frames)

# Separate harmonics and percussives into two waveforms
y_harmonic, y_percussive = librosa.effects.hpss(y)

# # calculate RMS for loudness/energy
S, phase = librosa.magphase(librosa.stft(y))
rms = librosa.feature.rms(S=S)

# print(rms)
# Set the hop length; at 22050 Hz, 512 samples ~= 23ms
hop_length = 512

# mel-scale for pitch
# mfccs = librosa.feature.mfcc(y, sr=sr)
# Compute MFCC features from the raw signal
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

# And the first-order differences (delta features)
mfcc_delta = librosa.feature.delta(mfcc)

# Stack and synchronize between beat events
# This time, we'll use the mean value (default) instead of median
beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]),
                                    beat_frames)

# Compute chroma features from the harmonic signal
chromagram = librosa.feature.chroma_cqt(y=y_harmonic,
                                        sr=sr)

# Aggregate chroma features between beat events
# We'll use the median value of each feature between beat frames
beat_chroma = librosa.util.sync(chromagram,
                                beat_frames,
                                aggregate=np.median)

# Finally, stack all beat-synchronous features together
beat_features = np.vstack([beat_chroma, beat_mfcc_delta])

