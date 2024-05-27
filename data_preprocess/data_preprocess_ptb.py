import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import pickle as cp
from data_preprocess.augmentations import gen_aug
import scipy.io
from scipy.signal import convolve
import matplotlib.pyplot as plt
from data_preprocess.base_loader import base_loader


def load_domain_data(domain_idx):
    mat = scipy.io.loadmat('data_preprocess/data/ptb_v3.mat')
    data = mat['data_to_save']
    data = data[0,int(domain_idx)] 
    raw_data = np.concatenate(data[:,0], axis=0) 
    #raw_data = data[:,0]
    bpms = data[:,1]
    return raw_data, bpms

class data_loader_ptb(base_loader):
    def __init__(self, samples, bpms, lin_ratio, args):
        super(data_loader_ptb, self).__init__(samples, bpms, lin_ratio, args)

    def __getitem__(self, index):
        sample, target, lin_ratio = self.samples[index], self.bpms[index], self.lin_ratio[index]
        sample = np.square(np.diff(sample,append=np.zeros((1,))))
        #sample = np.apply_along_axis(lambda x: convolve(x, np.ones(20) / 20, mode='same'), axis=0, arr=sample)
        #sample = (sample-np.min(sample))/(np.max(sample)-np.min(sample))
        sample = (sample-np.mean(sample))
        #sample = np.pad(sample, (100, 100), 'constant', constant_values=(0, 0))
        return torch.tensor(sample, device=self.args.cuda).float().unsqueeze(0), torch.tensor(target.item(),device=self.args.cuda).float(), lin_ratio
    
def collate_fn(batch):
    x_win, bpms, locs = np.array([]), np.zeros((len(batch),)), []
    for idx, i in enumerate(batch):
        x_win = np.concatenate((x_win, np.expand_dims(i[0],1)), axis=1) if x_win.size else np.expand_dims(i[0],1)
        bpms[idx] = i[1]
        locs.append(i[2])
    return torch.from_numpy(x_win).transpose(1,0).unsqueeze(1).float(), bpms, None

def prep_domains_ecg_ptb(args):
    xtrain, xbpms = load_domain_data('0')
    if args.augs:
        xtrain, xbpms, lin_ratio = aug_data(xtrain, xbpms, args)
    else: lin_ratio = np.ones((xtrain.shape[0], 1))
    xtest, xbpms_test = load_domain_data('1')

    data_set = data_loader_ptb(xtrain, xbpms, lin_ratio, args)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=None)    
    data_set_test = data_loader_ptb(xtest, xbpms_test, lin_ratio=np.ones((xtest.shape[0], 1)), args=args)
    target_loader = DataLoader(data_set_test, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=None)
    return source_loader, None, target_loader

def prep_domains_ecg_ptb_ssl_fn(args):
    xtrain, xbpms = load_domain_data('0')
    xtest, xbpms_test = load_domain_data('1')
    data_set = data_loader_ptb(xtrain, xbpms, lin_ratio=np.ones((xtrain.shape[0], 1)), args=args)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=None)    
    data_set_test = data_loader_ptb(xtest, xbpms_test, lin_ratio=np.ones((xtest.shape[0], 1)), args=args)
    target_loader = DataLoader(data_set_test, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=None)
    return source_loader, None, target_loader

def prep_domains_ecg_ptb_subject(args):
    xtrain, xbpms = load_domain_data('0')
    xtest, xbpms_test = load_domain_data('1')
    ###################################################### split the data into training and fine-tuning sets
    # Assuming xtrain and xbpms are your data tensors
    xtrain_shape = xtrain.shape[0]

    # Calculate the number of samples for fine-tuning set (10%)
    fine_tuning_size = int(0.10 * xtrain_shape)

    # Generate random indices for the fine-tuning set
    indices = np.arange(xtrain_shape)
    np.random.shuffle(indices)
    fine_tuning_indices = indices[:fine_tuning_size]
    # Split the data into training and fine-tuning sets
    xtrain_fine_tuning = xtrain[fine_tuning_indices]
    xbpms_fine_tuning = xbpms[fine_tuning_indices]
    #######################################################
    data_set = data_loader_ptb(xtrain_fine_tuning, xbpms_fine_tuning, lin_ratio=np.ones((xtrain_fine_tuning.shape[0], 1)), args=args)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=None)    
    data_set_test = data_loader_ptb(xtest, xbpms_test, lin_ratio=np.ones((xtest.shape[0], 1)), args=args)
    target_loader = DataLoader(data_set_test, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=None)
    return source_loader, None, target_loader

def prep_domains_ecg_ptb_subject_sp(args):
    xtrain, xbpms = load_domain_data('0')
    xtest, xbpms_test = load_domain_data('1')
    ###################################################### split the data into training and fine-tuning sets
    xtrain_shape = xtrain.shape[0]

    # Calculate the number of samples for fine-tuning set (10%)
    fine_tuning_size = int(0.10 * xtrain_shape)

    # Generate random indices for the fine-tuning set
    indices = np.arange(xtrain_shape)
    np.random.shuffle(indices)
    fine_tuning_indices = indices[:fine_tuning_size]
    train_indices = indices[0:-fine_tuning_size]
    # Split the data into training and fine-tuning sets
    xtrain_fine_tuning = xtrain[fine_tuning_indices]
    xbpms_fine_tuning = xbpms[fine_tuning_indices]
    xtrain = xtrain[train_indices]
    xbpms = xbpms[train_indices]
    #######################################################
    data_set = data_loader_ptb(xtrain_fine_tuning, xbpms_fine_tuning, lin_ratio=np.ones((xtrain_fine_tuning.shape[0], 1)), args=args)
    val_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=None)    

    data_set = data_loader_ptb(xtrain, xbpms, lin_ratio=np.ones((xtrain.shape[0], 1)), args=args)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=None)        

    data_set_test = data_loader_ptb(xtest, xbpms_test, lin_ratio=np.ones((xtest.shape[0], 1)), args=args)
    target_loader = DataLoader(data_set_test, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=None)
    return source_loader, val_loader, target_loader


def aug_data(xtrain, xbpms, args):
    num_samples = int(xtrain.shape[0] * args.augs_ratio)
    random_indices = np.random.choice(xtrain.shape[0], num_samples, replace=False)
    data_to_aug = xtrain[random_indices]

    data_to_aug_out = gen_aug(data_to_aug, args.aug_type, args)
    if isinstance(data_to_aug_out, tuple): 
        data_to_aug = data_to_aug_out[0]
        lin_ratio = data_to_aug_out[1]
    lin_ratio_orig = np.ones((xtrain.shape[0], 1))
    xtrain = np.concatenate((xtrain, data_to_aug), axis=0)
    xbpms = np.concatenate((xbpms, xbpms[random_indices]), axis=0)
    return xtrain, xbpms, np.concatenate((lin_ratio_orig, lin_ratio), axis=0)

def prep_ptb(args):
    if args.cases == 'subject_large':  # Non-contrastive --> No fine tuning
        return prep_domains_ecg_ptb(args)
    elif args.cases == 'subject':  # Fine tuning for contrastive
        return prep_domains_ecg_ptb_subject(args)
    elif args.cases == 'subject_large_ssl_fn':  # Pre-training for contrastive
        return prep_domains_ecg_ptb_ssl_fn(args)
    elif args.cases == 'subject_val':
        return prep_domains_ecg_ptb_subject_sp(args)    
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'