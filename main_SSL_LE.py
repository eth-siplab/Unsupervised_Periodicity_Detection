import os 
import argparse
import numpy as np
import random
from trainer_SSL_LE import *

parser = argparse.ArgumentParser(description='argument setting of network')
parser.add_argument('--cuda', default=3, type=int, help='cuda device ID: 0,1,2,3')
parser.add_argument('--seed', default=10, type=int, help='seed')
parser.add_argument('--optimizer', default='adam', type=str, choices=['adam','sgd'], help='optimizer')
parser.add_argument('--augs_ratio', default=0.2, type=float, help='Aug ratio')
parser.add_argument('--n_epoch', type=int, default=120, help='number of training epochs')
parser.add_argument('--out_dim', default=1000, type=float, help='out dim of the model')
parser.add_argument('--fs', default=100, type=float, help='sampling rate of the signal')
parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--scheduler', type=bool, default=True, help='if or not to use a scheduler')
parser.add_argument('--augs', action='store_true', help='Saving')  # Augmentation at the beginning (before pretraining)

# Augmentations
parser.add_argument('--aug1', type=str, default='jit_scal',
                    choices=['na', 'noise', 'scale', 'negate', 'perm', 'shuffle', 't_flip', 't_warp', 'resample', 'resample_2' , 'random_out', 'rotation', 'perm_jit', 'jit_scal', 'hfc', 'lfc', 'p_shift', 'ap_p', 'ap_f'],
                    help='the type of augmentation transformation')
parser.add_argument('--aug2', type=str, default='resample',
                    choices=['na', 'noise', 'scale', 'negate', 'perm', 'shuffle', 't_flip', 't_warp', 'resample', 'resample_2' , 'random_out', 'rotation', 'perm_jit', 'jit_scal', 'hfc', 'lfc', 'p_shift', 'ap_p', 'ap_f'],
                    help='the type of augmentation transformation')
parser.add_argument('--Randomfftmix', action='store_true', help='Saving') 

parser.add_argument('--dataset', default='ptb', choices=['dalia', 'ptb', 'wesad', 'ieee_small', 'ieee_big', 'clemson', 'clemson_semi', 'capno', 'capno_64', 'bidmc'], type=str, help='dataset name')
parser.add_argument('--overlap', default=2, type=float, help='overlap in seconds for the sliding window')
parser.add_argument('--downsample_ratio', default=5, type=float, help='Downsampling the original signal')
parser.add_argument('--data_type', default='ecg', choices=['ecg', 'imu_chest', 'imu_wrist', 'ppg'], type=str, help='data type')
parser.add_argument('--input_dim', default = 800, type=int, help='Input size of the original signal')
parser.add_argument('--lowest', default = 30, type=int, help='Lowest value of the label for the task, i.e., 30bpm')
parser.add_argument('--cases', type=str, default='subject_large_ssl_fn', choices=['subject', 'subject_large_ssl_fn'], help='name of scenarios, cross user')
parser.add_argument('--batch_size', default=512, type=int, help='batch size')
parser.add_argument('--test', action='store_true', help='test data')

# Frameworks
parser.add_argument('--framework', type=str, default='simclr', choices=['byol', 'simsiam', 'simclr', 'nnclr', 'tstcc', 'simper', 'vicreg', 'barlowtwins'], help='name of framework')
parser.add_argument('--lr_cls', type=float, default=1e-3, help='learning rate for linear classifier')
parser.add_argument('--backbone', type=str, default='DCL', choices=['FCN', 'DCL', 'DCL2', 'LSTM', 'AE', 'CNN_AE', 'Transformer', 'UNET', 'FCN2', 'RESNET'], help='name of backbone network')
parser.add_argument('--criterion', type=str, default='cos_sim', choices=['cos_sim', 'NTXent'],
                    help='type of loss function for contrastive learning')
parser.add_argument('--p', type=int, default=128,
                    help='byol: projector size, simsiam: projector output size, simclr: projector output size')
parser.add_argument('--phid', type=int, default=128,
                    help='byol: projector hidden size, simsiam: predictor hidden size, simclr: na')
# log
parser.add_argument('--logdir', type=str, default='log/', help='log directory')

# byol
parser.add_argument('--lr_mul', type=float, default=10.0,
                    help='lr multiplier for the second optimizer when training byol')
parser.add_argument('--EMA', type=float, default=0.996, help='exponential moving average parameter')

# nnclr
parser.add_argument('--mmb_size', type=int, default=1024, help='maximum size of NNCLR support set')

# TS-TCC
parser.add_argument('--lambda1', type=float, default=1.0, help='weight for temporal contrastive loss')
parser.add_argument('--lambda2', type=float, default=0.7, help='weight for contextual contrastive loss, also used as the weight for reconstruction loss when AE or CAE being backbone network')
parser.add_argument('--temp_unit', type=str, default='tsfm', choices=['tsfm', 'lstm', 'blstm', 'gru', 'bgru'], help='temporal unit in the TS-TCC')

# SimPer
parser.add_argument('--view_size', type=int, default=10, help='Number of variant views for SimPer')

#VicReg
parser.add_argument('--sim_coeff', type=float, default=25, help='weight for similarity loss')
parser.add_argument('--std_coeff', type=float, default=25, help='weight for standard deviation loss')
parser.add_argument('--cov_coeff', type=float, default=1, help='weight for covariance loss')

# Barlow Twins
parser.add_argument('--lambd', type=float, default=0.0051, help='weight for the off-diagonal terms in the covariance matrix')


# plot
parser.add_argument('--plt', type=bool, default=False, help='if or not to plot results')
parser.add_argument('--plot_tsne', action='store_true', help='if or not to plot t-SNE')

############ Example run ############
# python main_SSL_LE.py --framework 'simclr' --backbone 'FCN' --n_epoch 120 --batch_size 256 --lr 3e-3 --lr_cls 0.03 --cuda 3 --dataset 'ieee_small' --cases 'subject_large_ssl_fn' --aug1 'jit_scal' --aug2 'perm_jit' 

# Domains for each dataset
def set_domain(args):
    if args.framework == 'simper': args.batch_size = 64 # Low batch size for Simper
    if args.dataset == 'dalia':
        if args.data_type == 'ecg': # 8 seconds of ECG data
            args.out_dim = 640
            args.fs, args.lowest = 80, 30
            return [i for i in range(0, 15)]
        elif args.data_type == 'ppg': # 8 seconds of PPG data
            args.out_dim = 200
            args.fs, args.lowest = 25, 30
            return [i for i in range(0, 5)]
    elif args.dataset == 'ptb':  # 10 seconds of ECG data
        args.out_dim = 800
        args.fs, args.lowest = 80, 30
        return [i for i in range(0, 1)]
    elif args.dataset == 'wesad':
        if args.data_type == 'ecg': # 8 seconds of ECG data
            args.out_dim = 800
            args.fs, args.lowest = 100, 30
        elif args.data_type == 'ppg': # 8 seconds of PPG data
            args.out_dim = 200
            args.fs, args.lowest = 25, 30
        return [i for i in range(0, 15)]
    elif args.dataset == 'ieee_small': # 8 seconds of PPG data
        args.out_dim = 200
        args.fs, args.lowest, args.downsample_ratio = 25, 30, 5
        args.data_type = 'ppg'
        return [i for i in range(0, 12)]
    elif args.dataset == 'ieee_big':
        args.out_dim = 200
        args.fs, args.lowest, args.downsample_ratio = 25, 30, 5
        args.data_type = 'ppg'
        return [i for i in range(0, 22)][-6:]
    elif args.dataset == 'bidmc':
        args.out_dim = 800
        args.fs, args.lowest = 25, 5
        args.data_type = 'resp'
        return [i for i in range(0, 10)]
    elif args.dataset == 'capno' or args.dataset == 'capno_64':
        args.out_dim = 800 if args.dataset == 'capno' else 1600
        args.fs, args.lowest = 25, 4
        args.data_type = 'resp'
        args.batch_size = 228 if args.dataset == 'capno_64' else 256
        return [i for i in range(5, 10)]
    elif args.dataset == 'clemson' or args.dataset == 'clemson_semi':
        args.out_dim = 480
        args.fs, args.lowest = 15, 20
        args.data_type = 'step'
        return [i for i in range(0, 10)]

##################### Rep start #####################
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

############### Rep done ################
def main_SSL_LE(args):
    set_seed(args.seed)  # Change seed here
    setattr(args, 'cases', 'subject_large_ssl_fn') # Pretrain the models in the large unlabelled data 
    train_loaders, val_loader, test_loader = setup_dataloaders(args)
    model, optimizers, schedulers, criterion, classifier, criterion_cls, optimizer_cls = setup(args, DEVICE)

    best_pretrain_model = train(train_loaders, val_loader, model, DEVICE, optimizers, schedulers, criterion, args)

    best_pretrain_model = test(test_loader, best_pretrain_model, DEVICE, criterion, args)

    ############################################################################################################

    trained_backbone = lock_backbone(best_pretrain_model, args)  # Linear evaluation
    setattr(args, 'cases', 'subject') # Fine tune the models in the limited labelled data with the same target subject/domain
    train_loaders, val_loader, test_loader = setup_dataloaders(args)
    best_lincls = train_lincls(train_loaders, val_loader, trained_backbone, classifier, DEVICE, optimizer_cls, criterion_cls, args)
    error = test_lincls(test_loader, trained_backbone, best_lincls, DEVICE, criterion_cls, args, plt=args.plt)  # Evaluate with the target domain
    delete_files(args)
    return error

if __name__ == '__main__':
    args = parser.parse_args()
    set_seed(args.seed)
    DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print('device:', DEVICE, 'dataset:', args.dataset)
    whole_error = []
    for i in range(3):
        args.seed = np.random.randint(i*10,(i+1)*10)
        domain, error = set_domain(args), []
        for k in domain:
            print(f'Training for domain {k}')
            setattr(args, 'target_domain', k)    
            error.append(main_SSL_LE(args))
        error = np.asarray(error)
        whole_error.append([np.mean(error[:,0]), np.mean(error[:,1]), np.mean(error[:,2])])
        print(f'MAE: {np.mean(error[:,0])}, RMSE: {np.mean(error[:,1])}, r2: {np.mean(error[:,2])}\n')

    whole_error = np.asarray(whole_error)
    print(f'MAE: {np.mean(whole_error[:,0])}, RMSE: {np.mean(whole_error[:,1])}, r2: {np.mean(whole_error[:,2])}')
    print(f'Std MAE: {np.std(whole_error[:,0])}, Std RMSE: {np.std(whole_error[:,1])}, Std r2: {np.std(whole_error[:,2])}')
