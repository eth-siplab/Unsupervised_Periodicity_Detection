import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import scipy.io
import random
import pickle as cp
from data_preprocess.base_loader import base_loader
from data_preprocess.augmentations import gen_aug


def load_domain_data(domain_idx):
    str_folder = 'data_preprocess/data/'
    data_all = scipy.io.loadmat(str_folder + 'BIDMC.mat')
    data_all = data_all['whole_data']
    domain_idx = int(domain_idx)
    X = data_all[domain_idx,0].transpose()
    y = np.squeeze(data_all[domain_idx,1])
    return X, y

class data_loader_bidmc(base_loader):
    def __init__(self, samples, bpms, lin_ratio, args):
        super(data_loader_bidmc, self).__init__(samples, bpms, lin_ratio, args)

    def __getitem__(self, index):
        sample, target, lin_ratio = self.samples[index], self.bpms[index], self.lin_ratio[index]
 
        return torch.tensor(sample, device=self.args.cuda).float().unsqueeze(0), torch.tensor(target.item(),device=self.args.cuda).float(), lin_ratio


def prep_domains_bidmc_subject_large(args):
    source_domain_list = [str(i) for i in range(0, 53)]
    target_domain_list = [str(i) for i in range(args.target_domain*5, args.target_domain*5+5)]
    source_domain_list = [x for x in source_domain_list if x not in target_domain_list]
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

    data_set = data_loader_bidmc(xtrain, xbpms, lin_ratio, args)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    # target domain data prep
    xtest, ybpms = np.array([]), np.array([])
    for target_domain in target_domain_list:
        x, y = load_domain_data(target_domain)
        x = x.reshape((-1, x.shape[-1]))

        xtest = np.concatenate((xtest, x), axis=0) if xtest.size else x
        ybpms = np.concatenate((ybpms, y), axis=0) if ybpms.size else y

    data_set = data_loader_bidmc(xtest, ybpms, np.ones((xtest.shape[0], 1)), args)
    target_loader = DataLoader(data_set, batch_size=512, shuffle=False)  # For testing keep the batch size as 512

    return source_loader, None, target_loader

def prep_domains_bidmc_subject(args):
    source_domain_list = [str(i) for i in range(0, 53)]
    target_domain_list = [str(i) for i in range(args.target_domain*5, args.target_domain*5+5)]
    source_domain_list = [x for x in source_domain_list if x not in target_domain_list] 

    source_domain_list = random.sample(source_domain_list, 5) # 10 percent fo fine-tuning

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

    # # Identify unique classes and their counts
    # unique_classes, class_counts = np.unique(xbpms.round(), return_counts=True)

    # # Determine the minimum number of samples available for any class
    # min_samples_per_class = min(class_counts)

    # # Create an empty list to store the balanced signals
    # balanced_signals = []
    # balanced_classes = []
    # # Randomly sample the same number of samples from each class
    # for class_label in unique_classes:
    #     class_indices = np.where(xbpms.round() == class_label)[0]
    #     random_samples = random.sample(list(class_indices), min_samples_per_class)
    #     balanced_signals.extend(xtrain[random_samples])
    #     balanced_classes.extend(xbpms[random_samples])

    # # Determine the target number of samples you want in your balanced dataset
    # target_samples = 40  # Change this to your desired number

    # # Calculate how many more samples you need to reach the target
    # samples_needed = target_samples - len(balanced_signals)

    # if samples_needed > 0:
    #     # Randomly sample additional data from the source_domain
    #     for _ in range(samples_needed):
    #         random_source_domain = random.choice(source_domain_list)
    #         x, y = load_domain_data(random_source_domain)
    #         x = x.reshape((-1, x.shape[-1]))
            
    #         random_sample_index = random.randint(0, len(x) - 1)
    #         balanced_signals.append(x[random_sample_index])
    #         balanced_classes.append(y[random_sample_index])
    #         samples_needed -= 1

    # # Create a balanced signals array
    # balanced_signals_array = np.array(balanced_signals)

    data_set = data_loader_bidmc(xtrain, xbpms, lin_ratio, args)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    xtest, ybpms = np.array([]), np.array([])
    for target_domain in target_domain_list:
        x, y = load_domain_data(target_domain)
        x = x.reshape((-1, x.shape[-1]))

        xtest = np.concatenate((xtest, x), axis=0) if xtest.size else x
        ybpms = np.concatenate((ybpms, y), axis=0) if ybpms.size else y

    data_set = data_loader_bidmc(xtest, ybpms, np.ones((xtest.shape[0], 1)), args)
    target_loader = DataLoader(data_set, batch_size=512, shuffle=False)  # For testing keep the batch size as 512

    return source_loader, None, target_loader


def prep_domains_bidmc_subject_sp(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    # todo: make the domain IDs as arguments or a function with args to select the IDs (default, customized, small, etc)
    source_domain_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    
    source_domain_list.remove(args.target_domain)

    # source domain data prep
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        x, y, d = load_domain_data(source_domain)
        x = np.transpose(x.reshape((-1, 1, 200, 1)), (0, 2, 1, 3))

        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        d_win_all = np.concatenate((d_win_all, d), axis=0) if d_win_all.size else d

    x_win_train, x_win_val, \
    y_win_train, y_win_val, \
    d_win_train, d_win_val = train_val_split(x_win_all, y_win_all, d_win_all, split_ratio=args.split_ratio)

    unique_y, counts_y = np.unique(y_win_all, return_counts=True)
    weights = 100.0 / torch.Tensor(counts_y)
    weights = weights.double()

    sample_weights = get_sample_weights(y_win_all, weights)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights,
                                                             num_samples=len(sample_weights), replacement=True)

    data_set = data_loader_ieeesmall(x_win_train, y_win_train, d_win_train)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=None)
    source_loaders = [source_loader]

    ### 
    data_set_val = data_loader_ieeesmall(x_win_val, y_win_val, d_win_val)
    val_loader = DataLoader(data_set_val, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=None)
    val_loader = val_loader   

    # target domain data prep
    x, y, d = load_domain_data(args.target_domain)

    x = np.transpose(x.reshape((-1, 1, 200, 1)), (0, 2, 1, 3))


    data_set = data_loader_ieeesmall(x, y, d)
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    return source_loaders, val_loader, target_loader

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


def prep_bidmc(args):
    if args.cases == 'subject_large' or args.cases == 'subject_large_ssl_fn':
        return prep_domains_bidmc_subject_large(args)
    elif args.cases == 'subject_val':
        return prep_domains_bidmc_subject_sp(args)
    elif args.cases == 'subject':
        return prep_domains_bidmc_subject(args)
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'

