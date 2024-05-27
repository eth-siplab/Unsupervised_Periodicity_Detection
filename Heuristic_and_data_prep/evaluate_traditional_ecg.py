import scipy.io
from scipy.signal import resample
import WTdelineator as wav
import numpy as np
import pickle as cp
import matplotlib.pyplot as plt
from scipy.io import savemat

# https://github.com/caledezma/WTdelineator/tree/master


def take_fft(X, Fs, L):
    f = np.fft.fftfreq(L, 1 / Fs)[:L // 2]
    Y = np.fft.fft(X, L)
    P2 = abs(Y / L)
    P1 = P2[:L // 2]
    P1[1:-1] = 2 * P1[1:-1]
    f = f[1:]
    return f, P1

def find_closest_value(array, sample, fs, L=512):
    # Filter values within the range 30-210
    filtered_values = [value for value in array if 30 <= value <= 210]

    if not filtered_values:
        return 120  # No values within the specified range
    filtered_values = np.asarray(filtered_values)
    f,P1 = take_fft(sample, fs, L=L)
    bpms = f * 60
    closest_indices = np.abs(bpms - filtered_values[:, np.newaxis]).argmin(axis=1)
    max_index = np.argmax(P1[closest_indices])
    return filtered_values[max_index]

def find_closest_value_resp(array, sample, fs, L=512):
    # Filter values within the range 30-210
    filtered_values = [value for value in array if 40 <= value <= 140]

    if not filtered_values:
        return 60  # No values within the specified range
    filtered_values = np.asarray(filtered_values)
    f,P1 = take_fft(sample, fs, L=L)
    bpms = f * 60
    closest_indices = np.abs(bpms - filtered_values[:, np.newaxis]).argmin(axis=1)
    max_index = np.argmax(P1[closest_indices])
    return filtered_values[max_index]

def find_closest_value_2(array, real_bpm):

    closest_index = np.argmin(np.abs(array-real_bpm))
    return array[closest_index]

def evaluate_ptb():
    data_all = scipy.io.loadmat('ptb_v3.mat')
    data = data_all['data_to_save']
    data = data[0, int(1)]
    ppg = np.concatenate(data[:, 0], axis=0)
    bpms = data[:, 1]
    fs = 80
    # Target sample rate (1 kHz)
    target_sample_rate = 1000  # Hz
    # Calculate the resampling factor
    resampling_factor = target_sample_rate / fs
    # Calculate the new length of the resampled signal
    new_length = int(800 * resampling_factor)
    whole_error = []
    for k in range(len(ppg)):
        current_segment, segment_bpm = ppg[k, :], bpms[k]
        resampled_signal = resample(current_segment, new_length)
        Pwav, QRS, Twav = wav.signalDelineation(resampled_signal, int(target_sample_rate))
        if len(QRS) == 0:
            found_value = 60
        else:
            found_value = 60/(10/len(QRS))
        whole_error.append([found_value, segment_bpm.item()])
    return whole_error

############################################

def evaluate_wesad():
    data_all = scipy.io.loadmat('WESAD_ppg_ecg.mat')
    ecg = data_all['data_ecg']
    bpms = data_all['data_bpm_values']
    fs = 100
    # Target sample rate (1 kHz)
    target_sample_rate = 1000  # Hz
    # Calculate the resampling factor
    resampling_factor = target_sample_rate / fs
    # Calculate the new length of the resampled signal
    new_length = int(800 * resampling_factor)
    ##################################################
    whole_error = []
    for k in range(0,15):
        current_subject, subject_bpm = ecg[k, 0], bpms[k, 0]
        for l in range(len(current_subject)):
            resampled_signal = resample(current_subject[l,:], new_length)
            Pwav, QRS, Twav = wav.signalDelineation(resampled_signal, int(target_sample_rate))
            if len(QRS) == 0:
                found_value = 60
            else:
                first_elements = [sublist[0] for sublist in QRS]
                locs = np.array(first_elements)
                # found_value = np.mean(target_sample_rate/(np.diff(locs))*60)
                found_value = 60/(8/len(QRS))
            whole_error.append([found_value, subject_bpm[l].item()])
    return whole_error

########################

def evaluate_Dalia():
    str_folder = 'data_Dalia/'
    file = open(str_folder + 'dataDalia.pkl', 'rb')
    data = cp.load(file)
    fs = 80
    # Target sample rate (1 kHz)
    target_sample_rate = 1000  # Hz
    # Calculate the resampling factor
    resampling_factor = target_sample_rate / fs
    # Calculate the new length of the resampled signal
    new_length = int(800 * resampling_factor)
    whole_error = []
    for k in range(0,15):
        current_subject = data['data_ecg'][int(k)]
        # data_ppg = data['data_ppg'][int(domain_idx)]
        subject_bpm = data['data_bpm_labels'][int(k)]
        for l in range(len(current_subject)):
            resampled_signal = resample(current_subject[l,:], new_length)
            Pwav, QRS, Twav = wav.signalDelineation(resampled_signal, int(target_sample_rate))
            if len(QRS) == 0:
                found_value = 60
            else:
                found_value = 60/(8/len(QRS))
            whole_error.append([found_value, subject_bpm[l].item()])
    return whole_error

def error_to_np(error):
    error_array = np.asarray(error)
    MAE = np.mean(np.abs(error_array[:, 0] - error_array[:, 1]))
    RMSE = np.sqrt(np.mean(np.square(error_array[:, 0]-error_array[:, 1])))
    corr_coeff = np.corrcoef(error_array[:, 0], error_array[:, 1])[0,1]
    return MAE, RMSE, corr_coeff


if __name__ == '__main__':
    error_ptb = evaluate_ptb()
    metrics_ptb = error_to_np(error_ptb)

    error_dalia = evaluate_Dalia()
    metrics_dalia = error_to_np(error_dalia)

    error_wesad = evaluate_wesad()
    metrics_wesad = error_to_np(error_wesad)
    #############################################
    print(f'PTB MAE for wavelet based: {metrics_ptb[0]}')
    print(f'PTB RMSE for wavelet based: {metrics_ptb[1]}')
    print(f'PTB Corr for wavelet based: {metrics_ptb[2]}')
    ############################################
    print(f'Dalia MAE for wavelet based: {metrics_dalia[0]}')
    print(f'Dalia RMSE for wavelet based: {metrics_dalia[1]}')
    print(f'Dalia Corr for wavelet based: {metrics_dalia[2]}')
    #############################################
    print(f'Wesad MAE for wavelet based: {metrics_wesad[0]}')
    print(f'Wesad RMSE for wavelet based: {metrics_wesad[1]}')
    print(f'Wesad Corr for wavelet based: {metrics_wesad[2]}')
    print('exit')

