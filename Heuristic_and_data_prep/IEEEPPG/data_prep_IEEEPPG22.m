clear all
clc
close all
%%
addpath('C:\Users\bdemirel\Desktop\ETH Matlab\IEEEPPG\Competition_data\')
addpath('C:\Users\bdemirel\Desktop\ETH Matlab\berken_functions\')
%%
folders = dir('Competition_data\');
fs = 125;
window_duration = 8;
overlap_duration = 6;
data = {};
counter = 1;
for i = 3:12
current_data = load(folders(i).name).sig;
bpm_trace = load(folders(i+10).name).BPM0;
ppg1 = current_data(1,:);
ppg2 = current_data(2,:);
%
ppg1_filtered = filter_butter(ppg1,fs);
ppg2_filtered = filter_butter(ppg2,fs);
ppg_avg = (ppg1_filtered + ppg2_filtered)/2;
%
% ppg_avg_segments = downsample(normalize(buffer(ppg_avg,window_duration*fs,overlap_duration*fs),'zscore'),5).';
ppg_avg_segments = ((buffer(ppg_avg,window_duration*fs,overlap_duration*fs))).';
%%
data_ppg_avg{counter,1} = ppg_avg_segments(4:end-1,:);
if length(data_ppg_avg{counter,1}(:,1)) ~= length(bpm_trace)
data_ppg_avg{counter,1} = ppg_avg_segments(4:end,:);
end
data_bpm_values{counter,1} = bpm_trace;  
% Get SNR ratio
derived_snr = zeros((length(data_ppg_avg{counter,1}(:,1))),1);
for m = 1:length(data_ppg_avg{counter,1}(:,1))
    current_segment = data_ppg_avg{counter,1}(m,:);
    current_bpm = data_bpm_values{counter,1}(m);
    derived_snr(m) = give_snr(current_segment,current_bpm);
end
data_snr{counter,1} = derived_snr;  
counter = counter + 1;
end

save('IEEETest.mat','data_ppg_avg','data_bpm_values')
%% Filter Butterworth
function filtered = filter_butter(x,fs)
f1=0.5;
f2=4;
Wn=[f1 f2]*2/fs;
N = 3;
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
function derived_snr = give_snr(segment,bpm)
    L = 2048;
    fs = 25;
    Y = fft(segment,L);
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    P1 = normalize(P1,'range');
    f = fs*(0:(L/2))/L;
    bpm_indexes = f * 60;
    [~, indexOfMin] = min(abs(bpm_indexes-bpm));
    derived_snr = sum(P1(indexOfMin-7:indexOfMin+7))/sum(P1);
end