clear all 
clc
close all
%% Load data
data = load('matlab_ptb.mat'); 
xtrain = data.x_train;
xtest = data.x_test;
fs = 100;
%% Find peaks
whole_train = {};
whole_test = {};
bpm_test = zeros(length(xtest(:,1)),1);
locs_test = cell(length(xtest(:,1)),1);
for i = 1:length(xtest(:,1))
    current_data = xtest(i,:);
    [qrs_amp_raw,qrs_i_raw,delay]=pan_tompkin(current_data,fs,0);
    current_bpm = mean(fs./(diff(qrs_i_raw))*60);
    if current_bpm > 210 || isnan(current_bpm)
        bpm_test(i) = 0;
        locs_test{i,1} = 0;
    else
        bpm_test(i) = current_bpm;
        locs_test{i,1} = qrs_i_raw;
    end
end
%
bpm_train = zeros(length(xtrain(:,1)),1);
locs_train = cell(length(xtrain(:,1)),1);
for i = 1:length(xtrain(:,1))
    current_data = xtrain(i,:);
    [qrs_amp_raw,qrs_i_raw,delay]=pan_tompkin(current_data,fs,0);
    current_bpm = mean(fs./(diff(qrs_i_raw))*60);
    if current_bpm > 210 || isnan(current_bpm)
        bpm_train(i) = 0;
        locs_train{i,1} = 0;
    else
        bpm_train(i) = current_bpm;
        locs_train{i,1} = qrs_i_raw;
    end
end
%% Apply basic filtering
prep_xtrain = xtrain;
for i = 1:length(xtrain(:,1))
    current_data = xtrain(i,:);
    filtered_data = filter_butter(current_data,fs);
    whole_train{i,1} = resample(filtered_data,80,fs);
    whole_train{i,2} = bpm_train(i);
    whole_train{i,3} = locs_train{i};
end


prep_xtest = xtest;
for i = 1:length(xtest(:,1))
    current_data = xtest(i,:);
    filtered_data = filter_butter(current_data,fs);
    whole_test{i,1} = resample(current_data,80, fs);
    whole_test{i,2} = bpm_test(i);
    whole_test{i,3} = locs_test{i};
end
data_to_save = {};
data_to_save{1,1} = whole_train;
data_to_save{1,2} = whole_test;





%% Take fft
function [f,P1] = take_fft(X,Fs)
L = length(X);
f = Fs*(0:(L/2))/L;
Y = fft(X);
P2 = abs(Y/L);
P1 = P2(1:floor(L/2+1));
P1(2:end-1) = 2*P1(2:end-1);
P1 = P1(2:end);
f = f(2:end);
end
%% Filter Butterworth
function filtered = filter_butter(x,fs)
f1=0.7;
f2=40;
Wn=[f1 f2]*2/fs;
N = 4;
[b,a] = butter(N,Wn);
filtered = filtfilt(b,a,x);
filtered = normalize(filtered,'range');
end
