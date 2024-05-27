import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from scipy.io import savemat
import scipy.signal
from scipy.fft import rfft, rfftfreq
from scipy import stats

def assign_values(fs,L):
    xf = rfftfreq(L, d=1. / fs)
    loc_0, loc_1 = np.where((xf > 30/60))[0][0].item(), np.where((xf > 210/60))[0][0].item()  # 30 to 210 HR
    return loc_0, loc_1

def fft_bpm_method(x,fs, loc0, loc1, L, true_bpm=0):
    x = np.square(np.diff(x, append=[0]))
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    xf = rfftfreq(L, d=1./fs)
    bpms = 60*xf
    yr = np.abs(rfft(x, L))
    loc = np.argmax(yr[loc0:loc1])
    return bpms[loc+loc0]

def find_periodicity(signal, fs):
    x = np.square(np.diff(signal, append=[0]))
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    signal = x
    # Calculate the autocorrelation of the signal
    autocorrelation = np.correlate(signal, signal, mode='full')
    start_point = int(len(autocorrelation) // 2 + 1 + np.ceil(60*(fs/210)))
    end_point = int(len(autocorrelation) // 2 + 1 + np.ceil(60*(fs/30)))
    # Find the peak in the autocorrelation (excluding the zero-lag)
    peak_index = np.argmax(autocorrelation[start_point:end_point]) + 1
    peak_index = peak_index + start_point - len(autocorrelation) // 2 + 1
    # Calculate the periodicity
    periodicity = 60*(fs/peak_index)

    return periodicity


def give_snr(segment, bpm, fs, L):
    f, P1 = take_fft(segment, fs, L)
    bpms = 60 * f
    indexOfMin = np.argmin(abs(bpms-30))
    indexOfMax = np.argmin(abs(bpms-210))
    P1_modified = P1[indexOfMin:indexOfMax + 1]
    P1_modified = P1_modified / sum(P1_modified)

    bpm_index = np.argmin(np.abs(bpms - bpm))
    bpm_index -= indexOfMin
    if bpm_index <= 2:
        bpm_index = 3

    derived_snr = sum(P1_modified[bpm_index - 2:bpm_index + 3])

    return derived_snr

def take_fft(X, Fs, L):
    f = np.fft.fftfreq(L, 1 / Fs)[:L // 2]
    Y = np.fft.fft(X, L)
    P2 = abs(Y / L)
    P1 = P2[:L // 2]
    P1[1:-1] = 2 * P1[1:-1]
    f = f[1:]

    return f, P1


with open('dataDalia.pkl', 'rb') as pickle_file:
    data = pickle.load(pickle_file)

L = 2048
ecg, ecg_fs = data['data_ecg'], 80
ppg, ppg_fs = data['data_ppg'], 25
bpms = data['data_bpm_labels']
bpm_ecg_fft, ecg_se, snr_all = [], [], []
error_fft_ecg_mae, error_fft_ecg_rmse, error_fft_ecg_corr = [], [], []
error_fft_ppg_mae, error_fft_ppg_rmse, error_fft_ppg_corr = [], [], []
#
error_auto_ecg_mae, error_auto_ecg_rmse, error_auto_ecg_corr = [], [], []
error_auto_ppg_mae, error_auto_ppg_rmse, error_auto_ppg_corr = [], [], []
#
loc0_ppg, loc1_ppg = assign_values(ppg_fs,L)
loc0_ecg, loc1_ecg = assign_values(ecg_fs,L)
for i in range(15):
    subject_ecg, subject_ppg, subject_bpm = ecg[i], ppg[i], bpms[i]
    est_bpm_ecg_fft, est_bpm_ppg_fft = np.zeros((subject_ecg.shape[0],)), np.zeros((subject_ecg.shape[0],))
    est_bpm_ecg_auto, snr_subject = np.zeros((subject_ecg.shape[0],)), np.zeros((subject_ecg.shape[0],))
    est_bpm_ppg_auto = np.zeros((subject_ecg.shape[0],))
    for k in range(len(subject_ecg)):
        current_ecg, current_ppg, current_bpm = subject_ecg[k,:], subject_ppg[k,:], subject_bpm[k]
        est_bpm_ecg_fft[k] = fft_bpm_method(current_ecg, ecg_fs, loc0_ecg, loc1_ecg, L)
        est_bpm_ecg_auto[k] = find_periodicity(current_ecg, ecg_fs)
        est_bpm_ppg_fft[k] = fft_bpm_method(current_ppg, ppg_fs, loc0_ppg, loc1_ppg, L, current_bpm)
        est_bpm_ppg_auto[k] = find_periodicity(current_ppg, ppg_fs)
        snr_subject[k] = give_snr(current_ppg, current_bpm, ppg_fs, L)

    ecg_se.append(np.mean(np.abs(est_bpm_ecg_fft-subject_bpm)))
    # print(f'MAE (ECG) for subject {i} is {np.mean(np.abs(est_bpm_ecg_fft-subject_bpm))}')
    # print(f'MAE (PPG) for subject {i} is {np.mean(np.abs(est_bpm_ppg_auto - subject_bpm))}')
    bpm_ecg_fft.append(est_bpm_ecg_fft)
    # FFT ECG
    error_fft_ecg_mae.append(np.mean(np.abs(est_bpm_ecg_fft-subject_bpm)))
    error_fft_ecg_rmse.append(np.sqrt(np.mean((est_bpm_ecg_fft-subject_bpm)**2)))
    error_fft_ecg_corr.append(stats.pearsonr(est_bpm_ecg_fft, subject_bpm)[0])
    # FFT PPG
    error_fft_ppg_mae.append(np.mean(np.abs(est_bpm_ppg_fft-subject_bpm)))
    error_fft_ppg_rmse.append(np.sqrt(np.mean((est_bpm_ppg_fft-subject_bpm)**2)))
    error_fft_ppg_corr.append(stats.pearsonr(est_bpm_ppg_fft, subject_bpm)[0])
    # Autocorrelation ECG
    error_auto_ecg_mae.append(np.mean(np.abs(est_bpm_ecg_auto-subject_bpm)))
    error_auto_ecg_rmse.append(np.sqrt(np.mean((est_bpm_ecg_auto-subject_bpm)**2)))
    error_auto_ecg_corr.append(stats.pearsonr(est_bpm_ecg_auto, subject_bpm)[0])
    # Autocorrelation PPG
    error_auto_ppg_mae.append(np.mean(np.abs(est_bpm_ppg_auto-subject_bpm)))
    error_auto_ppg_rmse.append(np.sqrt(np.mean((est_bpm_ppg_auto-subject_bpm)**2)))
    error_auto_ppg_corr.append(stats.pearsonr(est_bpm_ppg_auto, subject_bpm)[0])
    #
    snr_all.append(snr_subject)
snr_all_conca = np.concatenate(snr_all)

print(f'Results for DaLiA')
print(f'--------------------------------------------------------------')
print(f'MAE (ECG FFT) overall is {np.mean(error_fft_ecg_mae)}')
print(f'RMSE (ECG FFT) overall is {np.mean(error_fft_ecg_rmse)}')
print(f'Correlation (ECG FFT) overall is {np.mean(error_fft_ecg_corr)}')
print(f'--------------------------------------------------------------')
#
print(f'MAE (ECG Auto) overall is {np.mean(error_auto_ecg_mae)}')
print(f'RMSE (ECG Auto) overall is {np.mean(error_auto_ecg_rmse)}')
print(f'Correlation (ECG Auto) overall is {np.mean(error_auto_ecg_corr)}')
print(f'--------------------------------------------------------------')
#
print(f'MAE (PPG FFT) overall is {np.mean(error_fft_ppg_mae)}')
print(f'RMSE (PPG FFT) overall is {np.mean(error_fft_ppg_rmse)}')
print(f'Correlation (PPG FFT) overall is {np.mean(error_fft_ppg_corr)}')
print(f'--------------------------------------------------------------')
#
print(f'MAE (PPG Auto) overall is {np.mean(error_auto_ppg_mae)}')
print(f'RMSE (PPG Auto) overall is {np.mean(error_auto_ppg_rmse)}')
print(f'Correlation (PPG Auto) overall is {np.mean(error_auto_ppg_corr)}')
print(f'--------------------------------------------------------------')

print('exit')