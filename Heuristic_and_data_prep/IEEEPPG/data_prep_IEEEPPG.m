clear all
clc
close all
%%
addpath('\...\Training_data\')
%%
folders = dir('Training_data\');
fs = 125;
window_duration = 8;
overlap_duration = 6;
data = {};
counter = 1;
for i = 3:2:25 % change according to the training data folder
current_data = load(folders(i).name).sig;
bpm_trace = load(folders(i+1).name).BPM0;
ecg = current_data(1,:);
ppg1 = current_data(2,:);
ppg2 = current_data(3,:);
acc1 = current_data(4,:);
acc2 = current_data(5,:);
acc3 = current_data(6,:);
%
ppg1_filtered = filter_butter(ppg1,fs);
ppg2_filtered = filter_butter(ppg2,fs);
ppg_avg = (ppg1_filtered + ppg2_filtered)/2;

ppg_avg_segments = downsample(normalize(buffer(ppg_avg,window_duration*fs,overlap_duration*fs),'zscore'),5).';
%%
data_ppg_avg{counter,1} = ppg_avg_segments(4:end-1,:);
% data_bpm_values{counter,1} = found_bpm(4:end-1,:);
if length(data_ppg_avg{counter,1}(:,1)) ~= length(bpm_trace)
data_ppg_avg{counter,1} = ppg_avg_segments(4:end,:);
end
data_bpm_values{counter,1} = bpm_trace;  
 
counter = counter + 1;
end
%% Evaluate
whole_error_fft_mae = [];
whole_error_fft_rmse = [];
whole_error_fft_corr = [];
whole_error_auto_mae = [];
whole_error_auto_rmse = [];
whole_error_auto_corr = [];
whole_snr = [];
fs = 25;
for i = 1:12
    current_subject = data_ppg_avg{i,1};
    current_bpm = data_bpm_values{i,1};
    subject_error_fft = [];
    subject_error_acf = [];
    subject_snr = [];
    for k = 1:length(current_subject(:,1))
        current_data = current_subject(k,:);
        predicted_bpm = predict_fft(current_data,fs);
        predicted_bpm_autocorr = predict_autocorr(current_data,fs);
        current_snr = give_snr(current_data,current_bpm(k), fs);
        subject_error_fft = [subject_error_fft ; predicted_bpm current_bpm(k)];
        subject_error_acf = [subject_error_acf ; predicted_bpm_autocorr current_bpm(k)];
        subject_snr = [subject_snr; current_snr abs(subject_error_fft(k,1)-subject_error_fft(k,2))];
    end
whole_snr = [whole_snr ; subject_snr(:,1)];
whole_error_fft_mae = [whole_error_fft_mae ;mean(abs(subject_error_fft(:,1)-subject_error_fft(:,2)))];
whole_error_fft_rmse = [whole_error_fft_rmse ;sqrt(mean((subject_error_fft(:,1)-subject_error_fft(:,2)).^2))];
whole_error_fft_corr = [whole_error_fft_corr ;unique(min(corrcoef(subject_error_fft(:,1),subject_error_fft(:,2))))];

whole_error_auto_mae = [whole_error_auto_mae ;mean(abs(subject_error_acf(:,1)-subject_error_acf(:,2)))];
whole_error_auto_rmse = [whole_error_auto_rmse ;sqrt(mean((subject_error_acf(:,1)-subject_error_acf(:,2)).^2))];
whole_error_auto_corr = [whole_error_auto_corr ;unique(min(corrcoef(subject_error_acf(:,1),subject_error_acf(:,2))))];
end
%
fprintf('MAE for FFT: %.3f\n',mean(whole_error_fft_mae))
fprintf('RMSE for FFT: %.3f\n',mean(whole_error_fft_rmse))
fprintf('corr for FFT: %.4f\n',mean(whole_error_fft_corr))
%
fprintf('MAE for ACF: %.3f\n',mean(whole_error_auto_mae))
fprintf('RMSE for ACF: %.3f\n',mean(whole_error_auto_rmse))
fprintf('corr for ACF: %.4f\n',mean(whole_error_auto_corr))
%%
function found_bpm = give_me_bpm(peak_locs,fs)
diff_locs = diff(peak_locs);
found_bpm = (fs ./ diff_locs)  * 60;
end
%% Filter Butterworth
function filtered = filter_butter(x,fs)
f1=0.5;
f2=4;
Wn=[f1 f2]*2/fs;
N = 4;
[b,a] = butter(N,Wn);
% [b,a] = ellip(6,5,50,20/(fs/2));
filtered = filtfilt(b,a,x);
filtered = normalize(filtered,'range');
end
%% Filter Butterworth
function filtered = filter_butter_ecg(x,fs)
     f1=3;                                                                      
     f2=20;                                                                     
     Wn=[f1 f2]*2/fs;                                                           
     N = 3;                                                                     
     [b,a] = butter(N,Wn);        
     % [b,a] = ellip(6,5,50,20/(fs/2));
     ecg_h = filtfilt(b,a,x);
     filtered = ecg_h/ max( abs(ecg_h));
end
%% ==================== SNR Derivation ========================== %%
function derived_snr = give_snr(segment,bpm,fs)
[f,P1] = take_fft(segment,fs);
bpms = 60*f;
[minDistance, indexOfMin] = min(abs(bpms-30));
[maxDistance, indexOfMax] = min(abs(bpms-210));
P1_modified = P1(indexOfMin:indexOfMax);
P1_modified = P1_modified.^2;
[~, bpm_index] = min(abs(bpms-bpm));
bpm_index = bpm_index - indexOfMin;
overall_power = sum(P1_modified);
signal_power = sum(P1_modified(bpm_index-2:bpm_index+2));
derived_snr = 10*log10(signal_power/(overall_power-signal_power));
end
%% Take fft
function [f,P1] = take_fft(X,Fs)
L = 512;
f = Fs*(0:(L/2))/L;
Y = fft(X,L);
P2 = abs(Y/L);
P1 = P2(1:floor(L/2+1));
P1(2:end-1) = 2*P1(2:end-1);
P1 = P1(2:end);
f = f(2:end);
end
%% Autocorr
function predicted_bpm = predict_autocorr(x,fs)
corrx = xcorr(x);
corrx = flip(corrx(1:length(x)));
[pks,locs] = findpeaks(corrx(ceil(60*(fs/210)):ceil(60*(fs/30))));
[M,I] = max(pks);
locs = locs(I);
predicted_loc = ceil(60*(fs/210)) + locs;
if ~ isnan(predicted_loc)
    predicted_bpm = 60*(fs/predicted_loc);
else
    predicted_bpm = 0;
end
end
%% FFT
function predicted_bpm = predict_fft(x,fs)
[f,P1] = take_fft(x,fs);
bpms = 60*f;
[minDistance, indexOfMin] = min(abs(bpms-30));
[maxDistance, indexOfMax] = min(abs(bpms-210));
[M,I] = max(P1(indexOfMin:indexOfMax));
predicted_loc = I + indexOfMin - 1;
if ~ isnan(predicted_loc)
    predicted_bpm = bpms(predicted_loc);
else
    predicted_bpm = 0;
end
end
