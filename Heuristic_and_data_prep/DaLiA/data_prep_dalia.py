import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import scipy.signal
from scipy import signal, stats
from scipy.fft import rfft, rfftfreq

sos = signal.butter(4, [0.5,4], btype='bandpass', fs=64, output='sos')
sos_imu = signal.butter(4, [0.5,4], btype='bandpass', fs=32, output='sos')
signal_list = []
label_list = []
activity_list = []
for file in os.listdir():
    d = os.path.join('', file)
    if os.path.isdir(d):
        print(d)
        with open(d+'/'+d+'.pkl', 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            ppg_signal = data['signal']['wrist']['BVP']
            
            filtered_ppg = signal.sosfilt(sos, ppg_signal)
            segments = np.lib.stride_tricks.sliding_window_view(np.squeeze(filtered_ppg), 64*8)[::128]
            z_scored = stats.zscore(segments,axis=1)
            resampled = scipy.signal.resample(z_scored, 200, axis=1)
            labels = data['label']
            signal_list.append(resampled)
            label_list.append(np.expand_dims(labels,1))
            ##
            segments_activity = np.lib.stride_tricks.sliding_window_view(data['activity'].squeeze(), 4 * 8)[::8]
            activity_list.append(segments_activity[:,0])

dataDalia = dict(data_ppg_avg=signal_list, data_bpm_values=label_list, data_activities=activity_list)
with open('Dalia_activity.pkl', 'wb') as handle:
    pickle.dump(dataDalia, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('exit')
