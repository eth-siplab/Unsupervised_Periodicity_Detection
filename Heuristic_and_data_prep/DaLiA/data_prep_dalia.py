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
ppg_fft_list = []
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
            ##
            # snr_array = np.zeros((len(resampled),))
            # for idx, i in enumerate(resampled):
            #     L = 2048
            #     P1 = rfft(i,L)
            #     P2 = np.abs(P1 / L)
            #     xf = rfftfreq(L, 1 / 25)
            #     bpm_indexes = xf * 60
            #     a = (np.abs(bpm_indexes - labels[idx])).argmin()
            #     derived_snr = np.sum(P2[a - 7:a + 7]) / np.sum(P2)
            #     snr_array[idx] = derived_snr
            L = 1024
            xf = np.fft.rfftfreq(L, d=1/25)
            xf_bpm = 60*xf
            min_index = min(range(len(xf_bpm)), key=lambda i: abs(xf_bpm[i]-30))
            max_index = min(range(len(xf_bpm)), key=lambda i: abs(xf_bpm[i]-210))
            xf_bpm = xf_bpm[min_index:max_index]
            ppg_fft = np.zeros((len(resampled), max_index-min_index,))
            for idx, i in enumerate(resampled):
                P1 = rfft(i,L)
                P1 = (1/(25*L)) * abs(P1)**2
                P1[1:-1] = 2*P1[1:-1]  # Multiply frequencies except DC and Nyquist
                P1 = P1/sum(P1)
                P1_bpm = P1[min_index:max_index]
                ppg_fft[idx] = P1_bpm
                if np.isnan(np.min(ppg_fft)):
                    print('gg')

            ppg_fft_list.append(ppg_fft)

dataDalia = dict(data_ppg_avg=signal_list, data_bpm_values=label_list, data_activities=activity_list)
with open('Dalia_activity.pkl', 'wb') as handle:
    pickle.dump(dataDalia, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('exit')