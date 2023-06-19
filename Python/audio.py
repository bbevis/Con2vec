

import os
import numpy as np
import moviepy.editor as mp
import ffmpeg
import librosa
import matplotlib.pyplot as plt

# os.environ["IMAGEIO_FFMPEG_EXE"] = "./.venv/lib/python3.10/site-packages/ffmpeg"
# load video
video = mp.VideoFileClip("Data/test_video.mov")

# extract the audio
audio = video.audio
audio.write_audiofile('./Data/audio.mp3')

print(audio)

y, sr = librosa.load('./Data/audio.mp3')

tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print(tempo)

# calculate RMS for loudness/energy
S, phase = librosa.magphase(librosa.stft(y))
rms = librosa.feature.rms(S=S)

print(rms)

# mel-scale for pitch
mfccs = librosa.feature.mfcc(y, sr=sr)
print(mfccs)

