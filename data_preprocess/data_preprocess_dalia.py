import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from scipy.signal import convolve
import pickle as cp
from data_preprocess.base_loader import base_loader_dalia
from data_preprocess.augmentations import gen_aug


def load_domain_data(domain_idx, args):
    str_folder = 'data_preprocess/data/'
    file = open(str_folder + 'dataDalia.pkl', 'rb')
    data = cp.load(file)

    z = data['data_activity'][int(domain_idx)]
    data_ecg = data['data_ecg'][int(domain_idx)]
    data_ppg = data['data_ppg'][int(domain_idx)]
    y = data['data_bpm_labels'][int(domain_idx)]
    if args.data_type == 'ecg': x = data_ecg
    elif args.data_type == 'ppg': x = data_ppg
    return x, y, z

class data_loader_dalia(base_loader_dalia):
    def __init__(self, samples, bpms, lin_ratio, activities, args):
        super(data_loader_dalia, self).__init__(samples, bpms, lin_ratio, activities, args)
        self.args = args

    def __getitem__(self, index):
        sample, target, locs, fs, lin_ratio = self.samples[index], self.bpms[index], self.activities[index], self.args.fs, self.lin_ratio[index]
        sample = np.square(np.diff(sample,append=1)) if self.args.data_type == 'ecg' else sample
        sample = (sample-np.min(sample))/(np.max(sample)-np.min(sample)) if self.args.data_type == 'ecg' else sample
        return torch.tensor(sample, device=self.args.cuda).float().unsqueeze(0), torch.tensor(target.item(),device=self.args.cuda).float(), lin_ratio, locs.astype(np.int16)

def collate_fn(batch):
    import pdb;pdb.set_trace();
    x_win, bpms, locs = np.array([]), np.zeros((len(batch),)), []
    for idx, i in enumerate(batch):
        x_win = np.concatenate((x_win, np.expand_dims(i[0],1)), axis=0) if x_win.size else np.expand_dims(i[0],1)
        bpms[idx] = i[1]
        locs.append(i[2])
    return torch.from_numpy(x_win).float(), bpms, np.array(locs)

def prep_domains_dalia_subject_large(args):
    source_domain_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
    source_domain_list.remove(str(args.target_domain))
    
    # source domain data prep
    x_win_all, y_win_all, z_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        #print('source_domain:', source_domain)
        x, y, z = load_domain_data(source_domain, args)

        x = x.reshape((-1, x.shape[-1]))

        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        z_win_all = np.concatenate((z_win_all, z), axis=0) if z_win_all.size else z

    if args.augs: x_win_all, y_win_all, lin_ratio, z_win_all, = aug_data(x_win_all, y_win_all, z_win_all, args)
    else: lin_ratio = np.ones((x_win_all.shape[0], 1))
 
    data_set = data_loader_dalia(x_win_all, y_win_all, lin_ratio, z_win_all, args)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=None)

    x, y, z = load_domain_data(args.target_domain, args)

    x = x.reshape((-1, x.shape[-1]))

    data_set = data_loader_dalia(x, y, np.ones((x.shape[0], 1)), z, args)
    target_loader = DataLoader(data_set, batch_size=1024, shuffle=False, collate_fn=None)

    return source_loader, None, target_loader

def prep_domains_dalia_subject(args):
    source_domain_list = ['0', '1', '2', '3', '4']
    if str(args.target_domain) in source_domain_list:
        source_domain_list.remove(str(args.target_domain))
    
    x_win_all, y_win_all, z_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        #print('source_domain:', source_domain)
        x, y, z = load_domain_data(source_domain, args)

        x = x.reshape((-1, x.shape[-1]))

        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        z_win_all = np.concatenate((z_win_all, z), axis=0) if z_win_all.size else z

    if args.augs: x_win_all, y_win_all, lin_ratio, z_win_all, = aug_data(x_win_all, y_win_all, z_win_all, args)
    else: lin_ratio = np.ones((x_win_all.shape[0], 1))
 
    data_set = data_loader_dalia(x_win_all, y_win_all, lin_ratio, z_win_all, args)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=None)

    x, y, z = load_domain_data(args.target_domain, args)

    x = x.reshape((-1, x.shape[-1]))

    data_set = data_loader_dalia(x, y, np.ones((x.shape[0], 1)), z, args)
    target_loader = DataLoader(data_set, batch_size=1024, shuffle=False, collate_fn=None)

    return source_loader, None, target_loader


def prep_domains_dalia_subject_sp(args):
    source_domain_list = ['0', '1', '2', '3', '4']
    source_domain_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
    if str(args.target_domain) in source_domain_list:
        source_domain_list.remove(str(args.target_domain))
    
    x_win_all, y_win_all, z_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        #print('source_domain:', source_domain)
        x, y, z = load_domain_data(source_domain, args)

        x = x.reshape((-1, x.shape[-1]))

        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        z_win_all = np.concatenate((z_win_all, z), axis=0) if z_win_all.size else z

    if args.augs: x_win_all, y_win_all, lin_ratio, z_win_all, = aug_data(x_win_all, y_win_all, z_win_all, args)
    else: lin_ratio = np.ones((x_win_all.shape[0], 1))

    ###################################################### split the data into training and validation sets
    # Assuming xtrain and xbpms are your data tensors
    xtrain_shape = x_win_all.shape[0]

    # Calculate the number of samples for validation set (10%)
    fine_tuning_size = int(0.1 * xtrain_shape)

    # Generate random indices for the validation set
    indices = np.arange(xtrain_shape)
    np.random.shuffle(indices)
    fine_tuning_indices = indices[:fine_tuning_size]
    training_indices = indices[fine_tuning_size:]
    # Split the data into training and validation sets
    xtrain_fine_tuning = x_win_all[fine_tuning_indices]
    xbpms_fine_tuning = y_win_all[fine_tuning_indices]
    zact_fine_tuning = z_win_all[fine_tuning_indices]
    # 
    xtrain_training = x_win_all[training_indices]
    xbpms_training = y_win_all[training_indices]    
    zact_training = z_win_all[training_indices]
    #######################################################
    data_set_val = data_loader_dalia(xtrain_fine_tuning, xbpms_fine_tuning, np.ones((xtrain_fine_tuning.shape[0], 1)), zact_fine_tuning, args)
    val_loader = DataLoader(data_set_val, batch_size=args.batch_size, shuffle=False)
    #
    data_set_train = data_loader_dalia(xtrain_training, xbpms_training, np.ones((xtrain_training.shape[0], 1)), zact_training, args)
    source_loader = DataLoader(data_set_train, batch_size=args.batch_size, shuffle=False)

    x, y, z = load_domain_data(args.target_domain, args)

    x = x.reshape((-1, x.shape[-1]))

    data_set = data_loader_dalia(x, y, np.ones((x.shape[0], 1)), z, args)
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, collate_fn=None)

    return source_loader, val_loader, target_loader

def aug_data(xtrain, xbpms, activities, args):
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
    activities = np.concatenate((activities, activities[random_indices]), axis=0)

    return xtrain, xbpms, np.concatenate((lin_ratio_orig, lin_ratio), axis=0), activities

def prep_dalia(args):
    if args.cases == 'subject_large' or args.cases == 'subject_large_ssl_fn':
        return prep_domains_dalia_subject_large(args)
    elif args.cases == 'subject':
        return prep_domains_dalia_subject(args)
    elif args.cases == 'subject_val':
        return prep_domains_dalia_subject_sp(args)     
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'