import os
import pandas as pd
print('working directory:')
print(os.getcwd())
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
# os.environ["IMAGEIO_FFMPEG_EXE"] = "./.venv/lib/python3.10/site-packages/ffmpeg"


import numpy as np
import ffmpeg
import librosa
import matplotlib.pyplot as plt
import scipy.io as sio


class vocal_features:
    
    def __init__(self, data):
        
        self.data = data

    def raw_outputs(self):

        y, sr = librosa.load(self.data)

        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
        
        # print(pulse)

        # Separate harmonics and percussives into two waveforms
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # tonal centroids
        # Compute the tonal centroid features (tonnetz)
        # This representation uses the method of 1 to project chroma features onto a 6-dimensional basis representing
        # the perfect fifth, minor third, and major third each as two-dimensional coordinates.
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        # print(tonnetz)

        # # calculate RMS for loudness/energy
        S, phase = librosa.magphase(librosa.stft(y))
        rms = librosa.feature.rms(S=S)
        # print(rms)

        # Set the hop length; at 22050 Hz, 512 samples ~= 23ms
        hop_length = 512

        # mel-scale for pitch
        # Compute Mel-frequency cepstral coefficients (MFCCs)features from the raw signal
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
        
        # Cluster by chroma similarity, break into 2 segments (speaking and not speaking)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        bounds = librosa.segment.agglomerative(chroma, 2)
        bound_times = librosa.frames_to_time(bounds, sr=sr)


        # Aggregate chroma features between beat events
        # We'll use the median value of each feature between beat frames
        beat_chroma = librosa.util.sync(chromagram,
                                        beat_frames,
                                        aggregate=np.median)

        # Finally, stack all beat-synchronous features together
        beat_features = np.vstack([beat_chroma, beat_mfcc_delta])

        Db = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        
        resAll = pd.DataFrame(
            {
            "tone_fifth_x": tonnetz[0],
            "tone_fifth_y": tonnetz[1],
            "tone_minor_x": tonnetz[2],
            "tone_minor_y": tonnetz[3],
            "tone_major_x": tonnetz[4],
            "tone_major_y": tonnetz[5],
            "loudness": rms[0],
            "pulse_tempo": pulse
        }
        )
        
        # resAll['tempo'] = tempo
        
        return resAll


if __name__ == '__main__':
    
    dirpath = 'Data_super_May22'
    group = '20240522_1325_S3WBLM9W4J66'
    # filename = '1710326137265-4144e390-caf9-40c5-9424-9cc5f734cbb6-cam-audio-1710326138270.wav'
    filename = '1716394969658-1a569948-8381-409b-9a04-fbb484b191a4-cam-audio-1716394970693'
    data =  os.path.join(dirpath, group, filename)
    # data =  os.path.join(dirpath, filename)
    # data =  '1716394969658-1a569948-8381-409b-9a04-fbb484b191a4-cam-audio-1716394970693.wav'
    
    vf = vocal_features(data)
    
    resAll = vf.raw_outputs()
    print(resAll)
    resAll.to_csv('Output/super_May22/test_output_vocal.csv', index=False)
    
    # fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    # D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    # img = librosa.display.specshow(D, y_axis='linear', x_axis='time',
    #                                sr=sr, ax=ax[0])
    # ax[0].set(title='Linear-frequency power spectrogram')
    # ax[0].label_outer()
    # D = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)),
    #                             ref=np.max)
    # librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=hop_length,
    #                          x_axis='time', ax=ax[1])
    # ax[1].set(title='Log-frequency power spectrogram')
    # ax[1].label_outer()
    # fig.colorbar(img, ax=ax, format="%+2.f dB")
    # plt.show()