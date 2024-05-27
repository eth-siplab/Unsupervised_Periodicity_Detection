import pickle 
import torch
import pywt
import ptwt

#######
# xfm = DWTForward(J=3, mode='zero', wave='db3').cuda()  # Accepts all wave types available to PyWavelets
# ifm = DWTInverse(mode='zero', wave='db3').cuda()
wavelet = pywt.Wavelet('haar')
####

file = open("data_preprocess/data/Dalia_chest.pkl",'rb')
data_dict = pickle.load(file)
file.close()
ecg_data = data_dict['data_ecg']
imu_data = data_dict['data_imu']
rpeaks = data_dict['data_rpeaks']
example = ecg_data[0]
example_ecg = example[0,:]
x_to_wavelet = torch.from_numpy(example_ecg)
x_unsqueeze = x_to_wavelet.float().cuda()
gg = ptwt.wavedec(x_unsqueeze, wavelet, mode='zero', level=2)
import pdb;pdb.set_trace();
reverse_gg = ptwt.waverec(gg, wavelet).squeeze()
import pdb;pdb.set_trace();