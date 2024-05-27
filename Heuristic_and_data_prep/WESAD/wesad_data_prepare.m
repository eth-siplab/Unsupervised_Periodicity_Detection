clear all
clc
close all
%%
data = load('WESAD_ppg');
ecg_signal = data.ecg_signal;
ppg_signal = data.ppg_signal;
window_duration = 8;
overlap_duration = 6;
ecg_fs = 700;
for i = 1:15
    current_ecg = ecg_signal{1,i};
    ecg_filtered = filter_butter_ecg(current_ecg, 700);
    ecg_segments = normalize(buffer(ecg_filtered,window_duration*ecg_fs,overlap_duration*ecg_fs),'zscore').';

    found_bpm = zeros(length(ecg_segments(1:end,1)),1);
    for k = 1:length(ecg_segments(:,1))
        data_to_tompkins = ecg_segments(k,:);
        [qrs_amp_raw,qrs_i_raw,delay] = pan_tompkin(data_to_tompkins,ecg_fs,0);
        found_bpm(k,1) = mean(ecg_fs./(diff(qrs_i_raw))*60);
        if isnan(found_bpm(k,1))
            found_bpm(k,1) = mean(found_bpm(k-4:k-1,1));
        end
    end
ecg_segments = resample(ecg_segments, 100, ecg_fs, Dimension=2);
data_bpm_values{i,1} = found_bpm(5:end-3,:); 
data_ecg{i,1} = ecg_segments(5:end-3,:);
end

save('WESAD_ecg.mat', 'data_ecg', 'data_bpm_values')

%% Evaluate
clear all
close all
clc
data = load('WESAD_ecg.mat');
data_ecg = data.data_ecg;
data_bpm_values = data.data_bpm_values;
fs = 100;
%%
whole_error_fft = [];
whole_error_auto = [];
for i = 1:15
    current_subject = data_ecg{i,1};
    current_bpm = data_bpm_values{i,1};
    subject_error_fft = [];
    subject_error_corr = [];
    for k = 1:length(current_subject(:,1))
        current_data = current_subject(k,:);
        current_data = movmean(diff(current_data).^2,1); 
        predicted_bpm = predict_fft(current_data,fs);
        predicted_bpm_autocorr = predict_autocorr(current_data,fs);
        subject_error_fft = [subject_error_fft ; predicted_bpm current_bpm(k)];
        subject_error_corr = [subject_error_corr ; predicted_bpm_autocorr current_bpm(k)];
    end

whole_error_fft = [whole_error_fft ;subject_error_fft(:,1) subject_error_fft(:,2)];
whole_error_auto = [whole_error_auto ;subject_error_corr(:,1) subject_error_corr(:,2)];
end

fprintf('MAE for FFT: %.3f\n',mean(abs(whole_error_fft(:,1)-whole_error_fft(:,2))))
fprintf('RMSE for FFT: %.3f\n',sqrt(mean((whole_error_fft(:,1)-whole_error_fft(:,2)).^2)))
fprintf('corr for FFT: %.3f\n',unique(min(corrcoef(subject_error_fft(:,1),subject_error_fft(:,2)))))
%
fprintf('MAE for ACF: %.3f\n',mean(abs(whole_error_auto(:,1)-whole_error_auto(:,2))))
fprintf('RMSE for ACF: %.3f\n',sqrt(mean((whole_error_auto(:,1)-whole_error_auto(:,2)).^2)))
fprintf('corr for ACF: %.3f\n',unique(min(corrcoef(whole_error_auto(:,1),whole_error_auto(:,2)))))


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
%% Take fft
function [f,P1] = take_fft(X,Fs)
L = 2048;
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
