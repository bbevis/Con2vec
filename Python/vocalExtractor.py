import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, lfilter
import scipy.fftpack
import librosa


print('working directory:')
print(os.getcwd())
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
# os.environ["IMAGEIO_FFMPEG_EXE"] = "./.venv/lib/python3.10/site-packages/ffmpeg"

import ffmpeg


class vocal_features:
    
    def __init__(self, data):
        
        self.data = data

    def raw_outputs(self):
        
        """
        Extracts as many useful vocal/prosodic features as possible using Librosa

        Returns:
            Timestamp: converts the frame indices into time stamps.
            Pitch (intonation patterns): returns an array of pitch values and corresponding magnitudes, selecting the pitch with the highest magnitude for each frame.
                The loop ensuring that only positive pitch values are considered (since negative or zero values are invalid for pitch). Can indicate questions, receptiveness
                and other vocal cues.
            RMS (Root Mean Square Energy): provides an estimate of the energy of the audio signal. A proxy for loudness/energy.
            Pulse (Onset Strength): estimates the strength of the onsets (a proxy for rhythm or pulse) in the audio signal.
                Rhythm and timing are crucial for understanding the cadence and structure of speech.
                It can reveal patterns in how speakers coordinate turn-taking and emphasize certain words or phrases.
            Zero-Crossing Rate (ZCR): measures how frequently the signal changes sign, which helps distinguish between different types of sounds, such as voiced vs. unvoiced speech.
            Spectral Centroid: Indicates where the center of mass of the spectrum is located.
            Spectral Bandwidth: Measures the width of the spectrum.
            Spectral Contrast: Captures the difference in amplitude between peaks and valleys in the spectrum.
            Tonnetz (Tonal Centroid Features): represents the tonal relations between notes in the audio, capturing the harmonic and melodic characteristics.
                Tonnetz features capture harmonic relations between pitches and can be used to study tonal aspects of speech, which may relate to emotional state or speech style.
            MFCCs (Mel-frequency Cepstral Coefficients): represent the short-term power spectrum of the audio signal, widely used in speech and audio processing.
            
        Synchronicity:
            Handling Length Mismatch: The code ensures that all features are synchronized by truncating them to the length of the shortest feature set.
            Data Storage: All extracted features are stored in a pandas DataFrame, and then saved as a CSV file.
        """

        y, sr = librosa.load(self.data)
        
        # Estimate pitch and magnitude from soundwave
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        
        # Compute additional features
        rms = librosa.feature.rms(y=y)[0]  # Root Mean Square Energy
        pulse = librosa.onset.onset_strength(y=y, sr=sr)  # Pulse (onset strength)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)  # Tonnetz features (tonal centroid)
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]  # Zero-Crossing Rate
        mfccs = librosa.feature.mfcc(y=y, sr=sr)  # MFCCs
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]  # Spectral Centroid
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]  # Spectral Bandwidth
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)  # Spectral Contrast
        
        # Compute durations and pauses
        non_silent_intervals = librosa.effects.split(y, top_db=20)
        durations = np.diff(librosa.frames_to_time(non_silent_intervals, sr=sr), axis=1).flatten()
        
        # Extract the timestamps for pitch, pulse, rms, and tonnetz
        time_stamps_pitch = librosa.frames_to_time(range(pitches.shape[1]), sr=sr)

        # Extract pitch values
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:  # Ignore non-positive values which are non-pitch
                pitch_values.append(pitch)
            else:
                pitch_values.append(np.nan)  # Use NaN for frames with no valid pitch

        # Ensure the lengths of feature arrays match
        min_length = min(len(rms), len(pulse), len(zcr), len(spectral_centroid), len(spectral_bandwidth), len(spectral_contrast[0]))
        print(min_length)
        
        df = pd.DataFrame(
            {
                "Time_ms": librosa.frames_to_time(range(min_length), sr=sr) * 1000,
                "Pitch_Hz": pitch_values[:min_length],
                "RMS": rms[:min_length],
                "Pulse": pulse[:min_length],
                "ZCR": zcr[:min_length],
                "Spectral_Centroid": spectral_centroid[:min_length],
                "Spectral_Bandwidth": spectral_bandwidth[:min_length]
            }
            )

        # Add Spectral Contrast as individual columns
        for i in range(spectral_contrast.shape[0]):
            df[f"Spectral_Contrast_{i+1}"] = spectral_contrast[i, :min_length]
            
        # Add Tonnetz features as individual columns
        for i in range(tonnetz.shape[0]):
            df[f"Tonnetz_{i+1}"] = tonnetz[i, :min_length]
            
        # Add MFCCs as individual columns
        for i in range(mfccs.shape[0]):
            df[f"MFCC_{i+1}"] = mfccs[i, :min_length]
            
        # Print the first 10 rows of the DataFrame
        # print(df.head(10))
        
        return df

if __name__ == '__main__':
    
    dirpath = 'Data_super_May22'
    group = '20240522_1325_S3WBLM9W4J66'
    filename = '1716394969658-1a569948-8381-409b-9a04-fbb484b191a4-cam-audio-1716394970693'
    data =  os.path.join(dirpath, group, filename)

    
    vf = vocal_features(data)
    resAll = vf.raw_outputs()
    # print(resAll)
    resAll.to_csv('Output/super_May22/test_output_vocal.csv', index=False)
    
   