import os
import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy
from torch import tensor
from torch.nn import Module
from torch.autograd import Function
from torch.distributions import Categorical
import torch.nn.functional as F
import torchvision
import torchaudio
from scipy import signal
import matplotlib.pyplot as plt
from data_preprocess import data_preprocess_dalia
from data_preprocess import data_preprocess_ptb
from data_preprocess import data_preprocess_wesad
from data_preprocess import data_preprocess_IEEE_small
from data_preprocess import data_preprocess_IEEE_big
from data_preprocess import data_preprocess_bidmc
from data_preprocess import data_preprocess_capno
from data_preprocess import data_preprocess_clemson
from losses import _IPR_SSL, _SNR_SSL, _EMD_SSL, resample_v2_loss, shiftfreq_loss

def setup_data_loader(args):
    if args.dataset == 'dalia':
        return data_preprocess_dalia.prep_dalia(args)
    elif args.dataset == 'ptb':
        return data_preprocess_ptb.prep_ptb(args)
    elif args.dataset == 'wesad':
        return data_preprocess_wesad.prep_wesad(args)
    elif args.dataset == 'ieee_small':
        return data_preprocess_IEEE_small.prep_ieee_small(args)
    elif args.dataset == 'ieee_big':
        return data_preprocess_IEEE_big.prep_ieeebig(args)
    elif args.dataset == 'bidmc':
        return data_preprocess_bidmc.prep_bidmc(args)
    elif args.dataset == 'capno' or args.dataset == 'capno_64':
        return data_preprocess_capno.prep_capno(args)
    elif args.dataset == 'clemson':
        return data_preprocess_clemson.prep_clemson(args)
    

class Trainer:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.model_name = 'models' + str(args.model) + args.dataset
        if self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            #self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        elif self.args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=15, factor=0.5, min_lr=1e-7, verbose=False)
        #self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=args.lr, max_lr=args.lr*10, cycle_momentum=False, step_size_up=100, step_size_down=100, mode='triangular2')

    def save_random(self, trainx, shiftedx, epoch, freq, labels):
        labels = labels.cpu().detach().numpy()
        L, loc0, loc1 = self.args.L, self.args.loc_0, self.args.loc_1
        bpms = freq.squeeze() * 60 if not self.args.data_type == 'step' else freq.squeeze() *60*(32/60)
        error_rate = np.abs(bpms-labels)
        indeces = np.argsort(error_rate)
        random_index = indeces[-10:]
        if not self.args.past_work1:
            epoch_folder = os.path.join('figures', 'epoch_' + str(self.args.dataset) + str(self.args.target_domain) + str(epoch))
        else:  epoch_folder = os.path.join('figures', 'epoch_' + str(self.args.dataset) + str(self.args.target_domain) + 'pw1' + str(epoch))
        os.makedirs(epoch_folder, exist_ok=True)
        for i in random_index:
            trainx_fft = torch.abs(torch.fft.rfft(trainx[i], n=L))
            shifted_fft = torch.abs(torch.fft.rfft(shiftedx[i], n=L))
            freq = self.args.fs*(torch.fft.rfftfreq(n=L))

            plt.plot(freq[loc0:loc1],trainx_fft[0,loc0:loc1].squeeze().cpu().detach().numpy())
            plt.plot(freq[loc0:loc1],shifted_fft[0,loc0:loc1].squeeze().cpu().detach().numpy())
            plt.vlines(labels[i]/60, 0, 200, colors='k', linestyles='dashed')
            plt.xticks(np.linspace(freq[loc0], freq[loc1], 10))
            plt.savefig(os.path.join(epoch_folder, str(i) + '.png'))
            plt.cla()
        return 0

    def evaluate_model(self, valid_loader, DEVICE, after_train=False):
        if after_train: 
            self.model.load_state_dict(torch.load('saved_models' + str(self.args.cuda) + '/' + str(self.args.model) + str(self.args.dataset) + str(self.args.data_type) + str(self.args.seed) +'.pt'))
        self.model.eval()
        avg_val_mse, avg_freq, avg_loss, total_samples, preds, true = 0., 0, 0, 0, [], []
        with torch.no_grad():
            for i, val_x in enumerate(valid_loader):
                filtered_signal, _ = self.model(val_x[0])

                if not self.args.past_work1:
                    filtered_signal = filtered_signal.unsqueeze(1)
                    loss, freq = self.freq_losses(filtered_signal, val_x[0], val_x[2])
                else:
                    loss, freq = self.freq_losses_2(filtered_signal, val_x[2])

                avg_loss += loss.detach().cpu().item() 
                avg_freq += torch.sum((freq*60).detach().cpu())
                if self.args.data_type == 'ecg' or self.args.data_type == 'ppg' or self.args.data_type == 'resp':
                    avg_val_mse += (torch.nn.L1Loss(reduction='sum')((freq*60).squeeze(), val_x[1]).item())
                elif self.args.data_type == 'step':
                    avg_val_mse += torch.sum((torch.abs((val_x[1] - freq*60*(32/60))/val_x[1])).squeeze()).item()
                else:
                    eval_points = val_x[3] == 1
                    if eval_points.any(): 
                        avg_val_mse += (torch.nn.L1Loss(reduction='sum')(freq[eval_points]*60, val_x[1][eval_points]).to(DEVICE)).item()
                total_samples += val_x[0].size(0)
                if self.args.data_type == 'step': preds.append(freq*60*(32/60))
                else: preds.append(freq*60)
                if after_train: true.append(val_x[1].cpu().detach().numpy())
    
        if after_train:
            preds, true = torch.cat(preds).cpu().detach().numpy(), np.concatenate(true)
            plt.scatter(preds, true)
            plt.savefig('figures' + '/' + 'eval_' + str(self.args.seed) + str(self.args.dataset) + str(self.args.target_domain) + '.png')
            plt.cla()
            if self.args.data_type == 'step': 
                print(f'MAPE: {np.mean(np.abs((true-preds)/true))}, MAE: {np.mean(np.abs(preds-true))}')
                return np.array([np.mean(np.abs((true-preds)/true)), 0, np.mean(np.abs(preds-true))])
            print(f'MSE: {np.mean(np.abs(preds-true))}, RMSE: {np.sqrt(np.mean(np.square(preds-true)))}, r2: {np.corrcoef(preds, true)[0,1]}')
            return np.array([np.mean(np.abs(preds-true)), np.sqrt(np.mean(np.square(preds-true))), np.corrcoef(preds, true)[0,1]])

        return avg_loss/total_samples
    
    def shift_tensors2(self, back_to_time, sample_shift, device_id):
        back_to_time = nn.ConstantPad1d(100,0)(back_to_time)
        fft_of_current = torch.fft.rfft(back_to_time, n=back_to_time.size(2))
        freq = torch.fft.rfftfreq(n=back_to_time.size(2)).cuda(self.args.cuda)
        fft_of_shifted = torch.zeros(fft_of_current.size(), requires_grad=True)
        ifft_of_shifted = torch.zeros(back_to_time.size(), requires_grad=True)
        fft_of_shifted = torch.exp(-1j*2*torch.pi*freq[None,:]*sample_shift[:,None])[:,None,:]*fft_of_current
        ifft_of_shifted = torch.fft.irfft(fft_of_shifted, back_to_time.size(2))
        ifft_of_shifted -= ifft_of_shifted.min(0, keepdim=True)[0]
        ifft_of_shifted /= ifft_of_shifted.max(0, keepdim=True)[0]
        ifft_of_shifted = ifft_of_shifted[:,:,100:-100]
        return ifft_of_shifted

    def freq_losses(self, filtered_signal, orig_signal, lin_ratio=None):
        L, loc_0, loc_1 = self.args.L, self.args.loc_0, self.args.loc_1

        x_fft = torch.abs(torch.fft.rfft(filtered_signal, n=L, norm='forward'))
        org_fft = torch.abs(torch.fft.rfft(orig_signal, n=L, norm='forward'))
        freq = self.args.fs*(torch.fft.rfftfreq(n=L).cuda(self.args.cuda))

        if self.args.aug_type == 'resample_v2':
            return resample_v2_loss(x_fft, org_fft, freq, loc_0, loc_1)
        if self.args.aug_type == 'freq_shift':
            return shiftfreq_loss(x_fft, org_fft, lin_ratio, loc_0, loc_1, freq, self.args)
        else:
            l1 = torch.sum((torch.sum(x_fft[:,:,0:loc_0], dim=2).squeeze() + torch.sum(x_fft[:,:,loc_1:], dim=2).squeeze()), dim=0)
            # Entropy
            freq_interest = x_fft[:,:,loc_0:loc_1]/torch.sum(x_fft[:,:,loc_0:loc_1], axis=2, keepdim=True)
            freq_interest_org = org_fft[:,:,loc_0:loc_1]/torch.sum(org_fft[:,:,loc_0:loc_1], axis=2, keepdim=True)

            l2 = torch.sum(-torch.sum(freq_interest*torch.log(freq_interest), dim=2), dim=0)
            #########
            kl_loss = nn.KLDivLoss(reduction='sum')
            l3 = kl_loss(torch.log(freq_interest), freq_interest_org)
            #########
            peak_locs = torch.argmax(x_fft[:,0, loc_0:loc_1], dim=1)
            
            return l1+l2+l3, freq[peak_locs+loc_0]

    def train(self, train_loader, valid_loader, split_seed, device_id):
        n_epoch, counter = 999, 0
        temp_avg_loss = np.inf
        for epoch in range(n_epoch):
            self.model.train()
            avg_loss, avg_freq, avg_mse, total_sample = 0, 0, 0, 0
            for i, train_x in enumerate(train_loader):
                filtered_signal, _ = self.model(train_x[0])
                
                if not self.args.past_work1:
                    filtered_signal = filtered_signal.unsqueeze(1)
                    loss, freq = self.freq_losses(filtered_signal, train_x[0], train_x[2])
                else:
                    loss, freq = self.freq_losses_2(filtered_signal, train_x[2])
                    filtered_signal = filtered_signal.unsqueeze(1)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.detach().cpu().item() 
                avg_freq += torch.sum((freq*60).detach().cpu())
                if self.args.data_type == 'ecg' or self.args.data_type == 'ppg':
                    avg_mse += (torch.nn.L1Loss(reduction='sum')((freq*60).squeeze(), train_x[1]).item())

                total_sample += train_x[0].size(0)

            if epoch % 40 == 0 and self.args.plot: self.save_random(train_x[0], filtered_signal, epoch, freq.detach().cpu(), train_x[1])
            avg_val_mse=self.evaluate_model(valid_loader, device_id)
            avg_loss, avg_freq, avg_mse = avg_loss/total_sample, avg_freq/total_sample, avg_mse/total_sample 

            if avg_loss + 0.001 < temp_avg_loss:
                counter = 0
                temp_avg_loss = avg_loss
                torch.save(deepcopy(self.model.state_dict()), 'saved_models' + str(self.args.cuda) + '/' 
                         + str(self.args.model) + str(self.args.dataset) + str(self.args.data_type) + str(self.args.seed) +'.pt')
            else:
                counter += 1
                if counter > 30:
                    return avg_loss

            self.scheduler.step(avg_loss)
        return avg_loss
    


###############################
    def freq_losses_2(self, filtered_signal, lin_ratio=None):
        L, loc_0, loc_1 = self.args.L, self.args.loc_0, self.args.loc_1

        x_fft = torch.abs(torch.fft.rfft(filtered_signal, n=L, norm='forward'))
        freq = self.args.fs*(torch.fft.rfftfreq(n=L).cuda(self.args.cuda))
        delta_freq = 0.1 

        if self.args.aug_type == 'resample_v2':
            l1_loss, l2_loss = torch.ones((x_fft.size(0),1)).cuda(self.args.cuda), torch.ones((x_fft.size(0),1)).cuda(self.args.cuda)
            l3_loss, bpms = torch.ones((x_fft.size(0),1)).cuda(self.args.cuda), torch.ones((x_fft.size(0),1)).cuda(self.args.cuda)
            for i in range(len(filtered_signal)):
                loc0, loc1 = (loc_0*lin_ratio[i]).int(), (loc_1*lin_ratio[i]).int()
                l1_loss[i] = _IPR_SSL(freq, x_fft[i,:], low_hz=freq[loc0], high_hz=freq[loc1], device=self.args.cuda)
                l2_loss[i] = _SNR_SSL(freq, x_fft[i,:], low_hz=freq[loc0], high_hz=freq[loc1], freq_delta=delta_freq, normalized=False, bandpassed=False, device=self.args.cuda)
                l3_loss[i] = _EMD_SSL(freq, x_fft[i,:], low_hz=freq[loc0], high_hz=freq[loc1], device=self.args.cuda)

            return torch.sum(l1_loss+l2_loss+l3_loss), bpms
        else:
            l1 = _IPR_SSL(freq, x_fft, low_hz=freq[loc_0], high_hz=freq[loc_1], device=self.args.cuda)
            l2 = _SNR_SSL(freq, x_fft, low_hz=freq[loc_0], high_hz=freq[loc_1], freq_delta=delta_freq, normalized=False, bandpassed=False, device=self.args.cuda)
            l3 = _EMD_SSL(freq, x_fft, low_hz=freq[loc_0], high_hz=freq[loc_1], device=self.args.cuda)

            peak_locs = torch.argmax(x_fft[:, loc_0:loc_1], dim=1)
            return l1+l2+l3, freq[peak_locs+loc_0]
        
###################################

def assign_fft_params(args):
    if args.data_type == 'ppg' or args.data_type == 'step': L = 512
    elif args.data_type == 'resp': L = 1024
    elif args.data_type == 'ecg': L = 2048
    else: L = 2048

    freq = args.fs*(torch.fft.rfftfreq(n=L).cuda(args.cuda))
    if args.data_type == 'resp': loc_0, loc_1 = torch.where((freq > 4/60))[0][0].item(), torch.where((freq > 40/60))[0][0].item() # 4 to 40 respiration
    elif args.data_type == 'step': loc_0, loc_1 = torch.where((freq > 40/60))[0][0].item(), torch.where((freq > 140/60))[0][0].item() # 40 to 140 step --> 20 to 60
    else: loc_0, loc_1 = torch.where((freq > 30/60))[0][0].item(), torch.where((freq > 210/60))[0][0].item() # 30 to 210 HR
    args.L = L
    args.loc_0 = loc_0
    args.loc_1 = loc_1
    return 
