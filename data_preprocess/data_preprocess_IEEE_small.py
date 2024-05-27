import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import scipy.io
import pickle as cp
from data_preprocess.base_loader import base_loader
from data_preprocess.augmentations import gen_aug


def load_domain_data(domain_idx, simper_aug=False):
    str_folder = 'data_preprocess/data/'
    data_all = scipy.io.loadmat(str_folder + 'IEEESmall_downsampled.mat') if not simper_aug else scipy.io.loadmat(str_folder + 'IEEESmall.mat') # Simper requires continuous data for augmentation
    ppg = data_all['data_ppg_avg']
    bpms = data_all['data_bpm_values']
    domain_idx = int(domain_idx)
    X = ppg[domain_idx,0]
    y = np.squeeze(bpms[domain_idx][0])
    return X, y

class data_loader_ieeesmall(base_loader):
    def __init__(self, samples, bpms, lin_ratio, args):
        super(data_loader_ieeesmall, self).__init__(samples, bpms, lin_ratio, args)

    def __getitem__(self, index):
        sample, target, lin_ratio = self.samples[index], self.bpms[index], self.lin_ratio[index]
        return torch.tensor(sample, device=self.args.cuda).float().unsqueeze(0), torch.tensor(target.item(),device=self.args.cuda).float(), lin_ratio


def prep_domains_ieeesmall_subject_large(args):
    source_domain_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    source_domain_list.remove(str(args.target_domain))
    
    # source domain data prep
    xtrain, xbpms, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        #print('source_domain:', source_domain)
        x, y = load_domain_data(source_domain, True if args.framework == 'simper' else False)

        x = x.reshape((-1, x.shape[-1]))

        xtrain = np.concatenate((xtrain, x), axis=0) if xtrain.size else x
        xbpms = np.concatenate((xbpms, y), axis=0) if xbpms.size else y
    
    if args.augs:
        xtrain, xbpms, lin_ratio = aug_data(xtrain, xbpms, args)
    else: lin_ratio = np.ones((xtrain.shape[0], 1))

    data_set = data_loader_ieeesmall(xtrain, xbpms, lin_ratio, args)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    x, y = load_domain_data(str(args.target_domain))

    x = x.reshape((-1, x.shape[-1]))

    data_set = data_loader_ieeesmall(x, y, np.ones((x.shape[0], 1)), args)
    target_loader = DataLoader(data_set, batch_size=512, shuffle=False)  # For testing keep the batch size as 512

    return source_loader, None, target_loader

def prep_domains_ieeesmall_subject(args):
    source_domain_list = ['0', '1', '2', '3', '4']
    if str(args.target_domain) in source_domain_list: source_domain_list.remove(str(args.target_domain))
    
    # source domain data prep
    xtrain, xbpms = np.array([]), np.array([])
    for source_domain in source_domain_list:
        #print('source_domain:', source_domain)
        x, y = load_domain_data(source_domain)
        x = x.reshape((-1, x.shape[-1]))

        xtrain = np.concatenate((xtrain, x), axis=0) if xtrain.size else x
        xbpms = np.concatenate((xbpms, y), axis=0) if xbpms.size else y
    
    if args.augs:
        xtrain, xbpms, lin_ratio = aug_data(xtrain, xbpms, args)
    else: lin_ratio = np.ones((xtrain.shape[0], 1))

    data_set = data_loader_ieeesmall(xtrain, xbpms, lin_ratio, args)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    x, y = load_domain_data(str(args.target_domain))

    x = x.reshape((-1, x.shape[-1]))

    data_set = data_loader_ieeesmall(x, y, np.ones((x.shape[0], 1)), args)
    target_loader = DataLoader(data_set, batch_size=512, shuffle=False)  # For testing keep the batch size as 512

    return source_loader, None, target_loader


def prep_domains_ieeesmall_subject_sp(args):
    source_domain_list = ['0', '1', '2', '3', '4']
    # source_domain_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    if str(args.target_domain) in source_domain_list: source_domain_list.remove(str(args.target_domain))
    
    # source domain data prep
    xtrain, xbpms = np.array([]), np.array([])
    for source_domain in source_domain_list:
        #print('source_domain:', source_domain)
        x, y = load_domain_data(source_domain)

        x = x.reshape((-1, x.shape[-1]))

        xtrain = np.concatenate((xtrain, x), axis=0) if xtrain.size else x
        xbpms = np.concatenate((xbpms, y), axis=0) if xbpms.size else y
    
    if args.augs:
        xtrain, xbpms, lin_ratio = aug_data(xtrain, xbpms, args)
    else: lin_ratio = np.ones((xtrain.shape[0], 1))

    ###################################################### split the data into training and fine-tuning sets
    # Assuming xtrain and xbpms are your data tensors
    xtrain_shape = xtrain.shape[0]

    # Calculate the number of samples for fine-tuning set (10%)
    fine_tuning_size = int(0.10 * xtrain_shape)

    # Generate random indices for the fine-tuning set
    indices = np.arange(xtrain_shape)
    np.random.shuffle(indices)
    fine_tuning_indices = indices[:fine_tuning_size]
    training_indices = indices[fine_tuning_size:]
    # Split the data into training and fine-tuning sets
    xtrain_fine_tuning = xtrain[fine_tuning_indices]
    xbpms_fine_tuning = xbpms[fine_tuning_indices]
    # 
    xtrain_training = xtrain[training_indices]
    xbpms_training = xbpms[training_indices]    
    #######################################################
    data_set_val = data_loader_ieeesmall(xtrain_fine_tuning, xbpms_fine_tuning, np.ones((xtrain_fine_tuning.shape[0], 1)), args)
    val_loader = DataLoader(data_set_val, batch_size=args.batch_size, shuffle=False)
    #
    data_set_train = data_loader_ieeesmall(xtrain_training, xbpms_training, np.ones((xtrain_training.shape[0], 1)), args)
    source_loader = DataLoader(data_set_train, batch_size=args.batch_size, shuffle=False)

    # Target domain data prep
    x, y = load_domain_data(str(args.target_domain))

    x = x.reshape((-1, x.shape[-1]))

    data_set = data_loader_ieeesmall(x, y, np.ones((x.shape[0], 1)), args)
    target_loader = DataLoader(data_set, batch_size=512, shuffle=False)  # For testing keep the batch size as 512

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


def prep_ieee_small(args):
    if args.cases == 'subject_large' or args.cases == 'subject_large_ssl_fn':
        return prep_domains_ieeesmall_subject_large(args)
    elif args.cases == 'subject_val':
        return prep_domains_ieeesmall_subject_sp(args)
    elif args.cases == 'subject':
        return prep_domains_ieeesmall_subject(args)
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'

