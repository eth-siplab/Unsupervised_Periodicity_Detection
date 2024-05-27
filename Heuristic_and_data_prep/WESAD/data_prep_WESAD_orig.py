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

mdic = {"ppg_signal": ppg_list,"ecg_signal":ecg_list}
savemat("WESAD_ppg.mat", mdic)
dataWESAD = dict(data_ecg_avg=ecg_list, data_ppg_avg=ppg_list)

print('exit')
