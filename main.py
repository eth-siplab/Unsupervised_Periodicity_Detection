import os 
import torch
import argparse
import random
import numpy as np
import functools
from models.models_nc import setup_model, count_parameters
from trainer import Trainer, setup_data_loader, assign_fft_params

parser = argparse.ArgumentParser(description='argument setting of network')
parser.add_argument('--cuda', default=3, type=int, help='cuda device ID: 0,1,2,3')
parser.add_argument('--seed', default=10, type=int, help='seed')
parser.add_argument('--past_work1', action='store_true', help='Saving')
parser.add_argument('--augs', action='store_true', help='Saving')
parser.add_argument('--optimizer', default='adam', type=str, choices=['adam','sgd'], help='optimizer')
parser.add_argument('--framework', default='unsup', type=str, choices=['unsup'], help='framework')
parser.add_argument('--augs_ratio', default=0.2, type=float, help='Aug ratio')
parser.add_argument('--out_dim', default=1000, type=float, help='out dim of the model')
parser.add_argument('--fs', default=100, type=float, help='sampling rate of the signal')
parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--aug_type', type=str, default='noise',
                    choices=['na', 'noise', 'scale', 'negate', 'perm', 'shuffle', 't_flip', 't_warp', 'resample', 'freq_shift', 'resample_2' , 'random_out','rotation', 'perm_jit', 'jit_scal', 'hfc', 'lfc', 'p_shift', 'ap_p', 'shift'],
                    help='the type of augmentation transformation')

parser.add_argument('--dataset', default='ptb', choices=['dalia', 'ptb', 'wesad', 'ieee_small', 'ieee_big', 'bidmc', 'capno', 'capno_64','clemson'], type=str, help='dataset name')
parser.add_argument('--data_type', default='ecg', choices=['ecg', 'imu_chest', 'imu_wrist', 'ppg','resp'], type=str, help='data type')
parser.add_argument('--input_dim', default = 800, type=int, help='Input size of the original signal')
parser.add_argument('--cases', type=str, default='subject_large', choices=['random', 'subject_large'], help='name of scenarios, cross user')
parser.add_argument('--model', type=str, default='unet', choices=['unet', 'resunet', 'convnet', 'dcl'], help='name of scenarios, cross user')
parser.add_argument('--batch_size', default=512, type=int, help='batch size')
parser.add_argument('--wandb', action='store_true', help='Saving')
parser.add_argument('--test', action='store_true', help='test data')
parser.add_argument('--plot', action='store_true', help='test data')

###########################
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_domain(args):
    if args.dataset == 'dalia':
        if args.data_type == 'ecg':
            args.out_dim = 640
            args.fs = 80
        elif args.data_type == 'ppg': # 8 seconds of PPG data with 64 Hz
            args.out_dim = 200
            args.fs = 25
        return [i for i in range(0, 15)]
    elif args.dataset == 'ptb':  # 10 seconds of ECG data
        args.out_dim = 800
        args.fs = 80
        return [i for i in range(0, 1)]
    elif args.dataset == 'wesad':
        if args.data_type == 'ecg':
            args.out_dim = 800
            args.fs = 100
        elif args.data_type == 'ppg':
            args.out_dim = 200
            args.fs = 25
        elif args.data_type == 'resp':
            args.out_dim = 200
            args.fs = 6
        return [i for i in range(0, 15)]
    elif args.dataset == 'ieee_small': # 8 seconds of PPG data
        args.out_dim = 200
        args.fs = 25
        args.data_type = 'ppg'
        return [i for i in range(0, 12)]
    elif args.dataset == 'ieee_big':
        args.out_dim = 200
        args.fs = 25
        args.data_type = 'ppg'
        return [i for i in range(0, 22)][-5:]
    elif args.dataset == 'bidmc':
        args.out_dim = 800
        args.fs = 25
        args.data_type = 'resp'
        return [i for i in range(0, 10)]
    elif args.dataset == 'capno' or args.dataset == 'capno_64':
        args.out_dim = 800 if args.dataset == 'capno' else 1600
        args.fs = 25
        args.data_type = 'resp'
        return [i for i in range(5, 10)]
    elif args.dataset == 'clemson':
        args.out_dim = 480
        args.fs = 15
        args.data_type = 'step'
        return [i for i in range(0, 10)]


############### Rep done ################

def train_func(args):
    domain, domain_error = set_domain(args), []
    assign_fft_params(args)
    for k in domain:
        print(f'Training for domain {k}')
        setattr(args, 'target_domain', k)        
        train_loaders, val_loader, test_loader = setup_data_loader(args)
        model = setup_model(args, DEVICE)

        trainer = Trainer(args, model)
        trainer.evaluate_model(test_loader, DEVICE)
        trainer.train(train_loaders, test_loader, args.seed, DEVICE)
        domain_error_saved = trainer.evaluate_model(test_loader, DEVICE, after_train=True)
        domain_error.append(domain_error_saved)
    return np.asarray(domain_error)


if __name__ == '__main__':
    args = parser.parse_args()
    set_seed(args.seed)
    DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print('device:', DEVICE)
    whole_error = []
    for i in range(3):
        set_seed(np.random.randint(i*10,(i+1)*10))
        error = train_func(args)
        whole_error.append([np.mean(error[:,0]), np.mean(error[:,1]), np.mean(error[:,2])])
        print(f'MAE: {np.mean(error[:,0])}, RMSE: {np.mean(error[:,1])}, r2: {np.mean(error[:,2])}')

    whole_error = np.asarray(whole_error)
    print(f'MAE: {np.mean(whole_error[:,0])}, RMSE: {np.mean(whole_error[:,1])}, r2: {np.mean(whole_error[:,2])}')
    print(f'Std MAE: {np.std(whole_error[:,0])}, Std RMSE: {np.std(whole_error[:,1])}, Std r2: {np.std(whole_error[:,2])}')
    
