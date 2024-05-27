import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import scipy.signal
from scipy import signal, stats
import heartpy as hp
from scipy.io import savemat

def read_csv_HR(d):
    list2 = []
    with open(d+'/'+'HR.csv') as f:
        for row in f:
            list2.append(row.split()[0])
    return np.array(list2[2:])

sos = signal.butter(4, [0.5,4], btype='bandpass', fs=64, output='sos')
ppg_list = []
ecg_list = []
imu_chest = []
imu_wrist = []
for file in os.listdir():
    d = os.path.join('', file)
    if os.path.isdir(d):
        print(d)
        with open(d+'/'+d+'.pkl', 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            ppg_signal = data['signal']['wrist']['BVP']
            ppg_list.append(ppg_signal)
            ecg = data['signal']['chest']['ECG']
            ecg_list.append(ecg)
            # imu_chest_sig = data['signal']['chest']['ACC']
            # imu_wrist_sig = data['signal']['wrist']['ACC']
            # total_acc = np.sqrt(imu_chest_sig[:,0]**2+imu_chest_sig[:,1]**2+imu_chest_sig[:,2]**2)
            # imu_chest.append(imu_chest_sig[:,1])
            # filtered_ppg = signal.sosfilt(sos, ppg_signal)
            # segments = np.lib.stride_tricks.sliding_window_view(np.squeeze(filtered_ppg), 64*8)[::128]
            # z_scored = stats.zscore(segments,axis=1)
            # resampled = scipy.signal.resample(z_scored, 200, axis=1)
            # HR = read_csv_HR(d)
            # HR = HR.astype(float)
            # HR_segmented = np.mean(np.lib.stride_tricks.sliding_window_view(HR, 1*8)[::2],1)
            # ecg = data['signal']['chest']['ECG']
            # ecg_segments = np.lib.stride_tricks.sliding_window_view(np.squeeze(ecg), 700*8)[::1400]
            # labels = np.zeros((len(segments)))
            # for idx in range(len(ecg_segments)):
            #     wd, m = hp.process(ecg_segments[idx, :], 700)
            #     if np.isnan(labels[idx]) or (idx > 3 and m['bpm'] > 25 + labels[idx-1]):
            #         labels[idx] = np.mean(labels[idx-4:idx])
            #     else:
            #         labels[idx] = m['bpm']
            # if len(np.argwhere(np.isnan(labels))) != 0:
            #     print('NAN value')
            # signal_list.append(resampled)
            # label_list.append(np.expand_dims(labels,1))

mdic = {"ppg_signal": ppg_list,"ecg_signal":ecg_list, 'resp_signal':resp_list}
savemat("WESAD_ppg.mat", mdic)
dataWESAD = dict(data_ecg_avg=ecg_list, data_ppg_avg=ppg_list, data_resp_signal=resp_list)

print('exit')
