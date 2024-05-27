clear all
clc
close all
%%
addpath('...\data_bidmc\')
%%
fs = 50;
window_duration = 8;
overlap_duration = 6;
data = load('bidmc_data.mat');
for i=1:53
current_ppg = data.data(i).ppg.v;
current_hr = data.data(i).ref.params.hr;
end
%%
fs = 25;
whole_error_fft_mae = [];
whole_error_fft_rmse = [];
whole_error_fft_corr = [];
whole_error_auto_mae = [];
whole_error_auto_rmse = [];
whole_error_auto_corr = [];
for i = 20:39
    current_subject = whole_data{i,1};
    current_bpm = whole_data{i,2};
    subject_error_fft = [];
    subject_error_acf = [];
    for k = 1:length(current_subject(1,:))
        current_data = current_subject(:,k);
        predicted_bpm = predict_fft(current_data,fs);
        predicted_bpm_autocorr = predict_autocorr(current_data,25);
        subject_error_fft = [subject_error_fft ; predicted_bpm current_bpm(k)];
        subject_error_acf = [subject_error_acf; predicted_bpm_autocorr current_bpm(k)];
    end
whole_error_fft_mae = [whole_error_fft_mae ;mean(abs(subject_error_fft(:,1)-subject_error_fft(:,2)))];
whole_error_fft_rmse = [whole_error_fft_rmse ;sqrt(mean((subject_error_fft(:,1)-subject_error_fft(:,2)).^2))];

whole_error_auto_mae = [whole_error_auto_mae ;mean(abs(subject_error_acf(:,1)-subject_error_acf(:,2)))];
whole_error_auto_rmse = [whole_error_auto_rmse ;sqrt(mean((subject_error_acf(:,1)-subject_error_acf(:,2)).^2))];
end

fprintf('MAE for FFT: %.3f\n',mean(whole_error_fft_mae))
fprintf('RMSE for FFT: %.3f\n',mean(whole_error_fft_rmse))
%
fprintf('MAE for ACF: %.3f\n',mean(whole_error_auto_mae))
fprintf('RMSE for ACF: %.3f\n',mean(whole_error_auto_rmse))


%% Take fft
function [f,P1] = take_fft(X,Fs)
L = 1024;
f = Fs*(0:(L/2))/L;
Y = fft(X,L);
P2 = abs(Y/L);
P1 = P2(1:floor(L/2+1));
P1(2:end-1) = 2*P1(2:end-1);
P1 = P1(2:end);
f = f(2:end);
end
%% Filter Butterworth
function filtered = filter_butter(x,fs)
     f1=0.05;                                                                      
     f2=0.6;                                                                     
     Wn=[f1 f2]*2/fs;                                                           
     N = 3;                                                                     
     [b,a] = butter(N,Wn);        
     % [b,a] = ellip(6,5,50,20/(fs/2));
     ecg_h = filter(b,a,x);
     filtered = ecg_h/ max( abs(ecg_h));
end
%% Autocorr
function predicted_bpm = predict_autocorr(x,fs)
corrx = xcorr(x);
corrx = flip(corrx(1:length(x)));
[pks,locs] = findpeaks(corrx(ceil(60*(fs/43)):ceil(60*(fs/5))));
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
[minDistance, indexOfMin] = min(abs(bpms-4));
[maxDistance, indexOfMax] = min(abs(bpms-40));
[M,I] = max(P1(indexOfMin:indexOfMax));
predicted_loc = I + indexOfMin - 1;
if ~ isnan(predicted_loc)
    predicted_bpm = bpms(predicted_loc);
else
    predicted_bpm = 0;
end
end

%%
function labels = give_me_labels(ecg_time_stamps, bpm_values)
last_second = ceil(ecg_time_stamps(end));
labels = [];
i = 0;
while i < last_second
ind_first = find(min(abs(ecg_time_stamps - i)) == abs(ecg_time_stamps - i));
ind_last = find(min(abs(ecg_time_stamps - (i+8))) == abs(ecg_time_stamps - (i+8)));
labels = [labels mean(bpm_values(ind_first:ind_last))];
i = i + 2;
end
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
