

import os
import magic
import numpy as np
# import moviepy.editor as mp
# import ffmpeg
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile

# os.environ["IMAGEIO_FFMPEG_EXE"] = "./.venv/lib/python3.10/site-packages/ffmpeg"
# load video
# video = mp.VideoFileClip("Data/test_video.mov")

# # extract the audio
# audio = video.audio
# audio.write_audiofile('./Data/audio.mp3')


dirpath = 'Data/super_icbs'
group = '20240312_1629_super_5KHZ83'
filename = '1710326137265-4144e390-caf9-40c5-9424-9cc5f734cbb6-cam-audio-1710326138270.wav'
filename_path =  os.path.join(dirpath, group, filename)

# dat, sampleRate = librosa.load(f'{filename_path}.wav')

# wav_data = wavfile.read(filename_path)


# y, sr = librosa.load(filename_path)

# tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
# print(tempo)

# print(audio)

# y, sr = librosa.load('./Data/audio.mp3')

# tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
# print(tempo)

# # calculate RMS for loudness/energy
# S, phase = librosa.magphase(librosa.stft(y))
# rms = librosa.feature.rms(S=S)

# print(rms)

# # mel-scale for pitch
# mfccs = librosa.feature.mfcc(y, sr=sr)
# print(mfccs)

