# encoding=utf-8
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import confusion_matrix
from models.backbones import *
from models.models_nc import ResNet1D, multi_rate
from models.loss import *
from trainer import *
import torch
import torch.nn as nn
import argparse
import random
from datetime import datetime
import pickle
import numpy as np
import os
import logging
import sys
from trainer_SSL_LE import setup_dataloaders
from scipy import signal
from copy import deepcopy
import fitlog
from new_augmentations import vanilla_mixup_sup, cutmix_sup
# fitlog.debug()


parser = argparse.ArgumentParser(description='argument setting of network')
parser.add_argument('--cuda', default=0, type=int, help='cuda device ID, 0/1')
# hyperparameter
parser.add_argument('--batch_size', type=int, default=64, help='batch size of training')
parser.add_argument('--n_epoch', type=int, default=60, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lr_cls', type=float, default=1e-3, help='learning rate for linear classifier')
parser.add_argument('--VAE', action='store_true')
parser.add_argument('--VanillaMixup', action='store_true')
parser.add_argument('--BinaryMix', action='store_true')
parser.add_argument('--Cutmix', action='store_true')
parser.add_argument('--Magmix', action='store_true')
parser.add_argument('--augs', action='store_true')
parser.add_argument('--regress', action='store_true')
# dataset
parser.add_argument('--dataset', default='ptb', choices=['dalia', 'ptb', 'wesad', 'ieee_small', 'ieee_big','capno','capno_64', 'clemson'], type=str, help='dataset name')
parser.add_argument('--data_type', default='ppg', choices=['ecg', 'imu_chest', 'ppg'], type=str, help='data type')
parser.add_argument('--lowest', default = 30, type=int, help='Lowest frequency of the original signal')
parser.add_argument('--n_feature', type=int, default=77, help='name of feature dimension')
parser.add_argument('--len_sw', type=int, default=30, help='length of sliding window')
parser.add_argument('--n_class', type=int, default=18, help='number of class')
parser.add_argument('--cases', type=str, default='subject_val', choices=['random', 'subject', 'subject_large', 'cross_device', 'joint_device'], help='name of scenarios')
parser.add_argument('--split_ratio', type=float, default=0.2, help='split ratio of test/val: train(0.64), val(0.16), test(0.2)')
parser.add_argument('--target_domain', type=str, default='0', help='the target domain, [0 to 29] for ucihar, '
                                                                   '[1,2,3,5,6,9,11,13,14,15,16,17,19,20,21,22,23,24,25,29] for shar, '
                                                                   '[a-i] for hhar')

# backbone model
parser.add_argument('--backbone', type=str, default='DCL', choices=['FCN', 'FCN2', 'DCL', 'cnn_lstm', 'LSTM', 'AE', 'CNN_AE', 'Transformer', 'UNET', 'resnet', 'multirate'], help='name of framework')

# log
parser.add_argument('--logdir', type=str, default='log/', help='log directory')

# AE & CNN_AE
parser.add_argument('--lambda1', type=float, default=1.0, help='weight for reconstruction loss when backbone in [AE, CNN_AE]')

# python main_supervised_baseline.py --dataset 'ecg' --cuda 1

############### Parser done ################

def train(args, train_loaders, val_loader, model, DEVICE, optimizer, criterion, save_dir='results/'):
    min_val_loss = 1e8
    for epoch in range(args.n_epoch):
        train_loss = 0
        n_batches = 0
        total = 0
        correct = 0
        model.train()
        for idx, train_x in enumerate(train_loaders):
            sample, target = train_x[0], train_x[1]
            n_batches += 1
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
            target = target.long() - args.lowest  # 30 -- 210 bpm to 0 -- 180 class
            target = torch.clamp(target, min=0)            
            if args.backbone[-2:] == 'AE':
                out, x_decoded = model(sample)
            else:
                sample = sample.transpose(2,1)
                if args.backbone == 'multirate':
                    out, regus = model(sample)
                else:
                    out, _ = model(sample)

            loss = criterion(out, target) if args.n_class > 1 else criterion(out.squeeze(1), target.float())

            if args.backbone[-2:] == 'AE':
                # print(loss.item(), nn.MSELoss()(sample, x_decoded).item())
                loss += nn.MSELoss()(sample, x_decoded) * args.lambda1

            if args.backbone == 'multirate':
                loss += sum(regus)

            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if val_loader is None:
            best_model = deepcopy(model.state_dict())
            model_dir = save_dir + args.model_name + '.pt'
            # print('Saving models at {} epoch to {}'.format(epoch, model_dir))
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, model_dir)
        else:
            with torch.no_grad():
                model.eval()
                val_loss = 0
                n_batches = 0
                total = 0
                correct = 0
                for idx, train_x in enumerate(train_loaders):
                    sample, target = train_x[0], train_x[1]
                    n_batches += 1
                    sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
                    target = target.long() - args.lowest  
                    target = torch.clamp(target, min=0)
                    if args.backbone[-2:] == 'AE':
                        out, x_decoded = model(sample)
                    else:
                        sample = sample.transpose(2,1)
                        out, _ = model(sample)

                    loss = criterion(out, target) if args.n_class > 1 else criterion(torch.squeeze(out), target.float())
                    if args.backbone[-2:] == 'AE':
                        loss += nn.MSELoss()(sample, x_decoded) * args.lambda1
                    val_loss += loss.item()
                    _, predicted = torch.max(out.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum()
                if val_loss <= min_val_loss:
                    min_val_loss = val_loss
                    best_model = deepcopy(model.state_dict())
                    # print('update')
                    model_dir = save_dir + args.model_name + '.pt'
                    # print('Saving models at {} epoch to {}'.format(epoch, model_dir))
                    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, model_dir)

    return best_model

def test(args, test_loader, model, DEVICE, criterion, plt=False):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        n_batches = 0
        total = 0
        correct = 0
        feats = None
        prds = None
        trgs = None
        otp = np.array([])
        confusion_matrix = torch.zeros(args.n_class, args.n_class)
        for idx, train_x in enumerate(test_loader):
            sample, target = train_x[0], train_x[1]
            n_batches += 1
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
            sample = sample.transpose(2,1)
            target = target.long() - args.lowest  
            target = torch.clamp(target, min=0)
            out, features = model(sample)
            loss = criterion(out, target) if args.n_class > 1 else criterion(torch.squeeze(out), target.float())
            total_loss += loss.item()
            _, predicted = torch.max(out.data, 1) if args.n_class > 1 else (0, out.squeeze().data)
            if prds is None:
                prds = predicted
                trgs = target
                # feats = features[:, :]
            else:
                prds = torch.cat((prds, predicted))
                trgs = torch.cat((trgs, target))
                # feats = torch.cat((feats, features), 0)
        
    if args.dataset == 'clemson':
        trgs, prds = trgs + args.lowest, prds + args.lowest
        acc_test = 100*torch.mean(torch.abs((trgs-prds)/trgs)).cpu()
        maF = torch.mean((torch.abs(trgs-prds)).float()).cpu()
        mr = 1
    else:
        maF = np.sqrt(torch.mean(((trgs-prds)**2).float()).cpu())
        acc_test = torch.mean((torch.abs(trgs-prds)).float()).cpu()
        mr = np.corrcoef(prds.detach().cpu(), trgs.detach().cpu())[0,1]
    if np.isnan(mr): mr = 0
    return acc_test, maF, mr

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

def train_sup(args, i):
    set_seed(np.random.randint(i*10,(i+1)*10))
    DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')

    train_loaders, val_loader, test_loader = setup_dataloaders(args)

    if args.regress: args.n_class = 1

    if args.backbone == 'FCN':
        model = FCN(n_channels=args.n_feature, in_dim=args.out_dim, n_classes=args.n_class, backbone=False)
    elif args.backbone == 'FCN2':
        model = FCN_2(n_channels=args.n_feature, in_dim=args.out_dim, n_classes=args.n_class, backbone=False)
    elif args.backbone == 'DCL':
        model = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=False, regress=args.regress)
    elif args.backbone == 'DCL2':
        model = DeepConvLSTM_2(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=7, LSTM_units=128, backbone=False)
    elif args.backbone == 'cnn_lstm':
        model = cnn_lstm(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, regress=args.regress)
    elif args.backbone == 'LSTM':
        model = LSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=128, backbone=False)
    elif args.backbone == 'AE':
        model = AE(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, outdim=128, backbone=False)
    elif args.backbone == 'CNN_AE':
        model = CNN_AE(n_channels=args.n_feature, n_classes=args.n_class, out_channels=128, backbone=False)
    elif args.backbone == 'Transformer':
        model = Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=False)
    elif args.backbone == 'UNET':
        model = UNET_1D_simp_ssl(input_dim=1, output_dim=args.out_dim, layer_n=32, kernel_size=5, depth=1, args=args, backbone=False, n_class=args.n_class)
    elif args.backbone == 'resnet':
        model = ResNet1D(in_channels=args.n_feature, base_filters=32, kernel_size=5, stride=2, groups=1, n_block=8, n_classes=args.n_class, downsample_gap=2, increasefilter_gap=4, backbone=False)
    elif args.backbone == 'multirate':
        model = multi_rate(num_classes=args.n_class, first_conv=args.n_feature, conv_kernels=16, backbone=False)
    else:
        NotImplementedError

    model = model.to(DEVICE)

    # print('Number of parameters: ', sum(p.numel() for p in model.parameters()))

    args.model_name = args.backbone + '_'+args.dataset + '_lr' + str(args.lr) + '_bs' + str(args.batch_size) + '_sw' + str(args.len_sw)

    save_dir = 'results/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # log
    if os.path.isdir(args.logdir) == False:
        os.makedirs(args.logdir)
    log_file_name = os.path.join(args.logdir, args.model_name + f".log")

    criterion = nn.CrossEntropyLoss() if args.n_class > 1 else nn.L1Loss()

    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, args.lr)

    best_model = train(args, train_loaders, val_loader, model, DEVICE, optimizer, criterion)

    if args.backbone == 'FCN':
        model_test = FCN(n_channels=args.n_feature, in_dim=args.out_dim, n_classes=args.n_class, backbone=False)
    elif args.backbone == 'FCN2':
        model_test = FCN_2(n_channels=args.n_feature, in_dim=args.out_dim, n_classes=args.n_class, backbone=False)
    elif args.backbone == 'DCL':
        model_test = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=False, regress=args.regress)
    elif args.backbone == 'DCL2':
        model_test = DeepConvLSTM_2(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=7, LSTM_units=128, backbone=False)
    elif args.backbone == 'cnn_lstm':
        model_test = cnn_lstm(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, regress=args.regress)        
    elif args.backbone == 'LSTM':
        model_test = LSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=128, backbone=False)
    elif args.backbone == 'AE':
        model_test = AE(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, outdim=128, backbone=False)
    elif args.backbone == 'CNN_AE':
        model_test = CNN_AE(n_channels=args.n_feature, n_classes=args.n_class, out_channels=128, backbone=False)
    elif args.backbone == 'Transformer':
        model_test = Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=False)
    elif args.backbone == 'UNET':
        model_test = UNET_1D_simp_ssl(input_dim=1, output_dim=args.out_dim, layer_n=32, kernel_size=5, depth=1, args=args, backbone=False, n_class=args.n_class)
    elif args.backbone == 'resnet':
        model_test = ResNet1D(in_channels=args.n_feature, base_filters=32, kernel_size=5, stride=2, groups=1, n_block=8, n_classes=args.n_class, downsample_gap=2, increasefilter_gap=4, backbone=False)
    elif args.backbone == 'multirate':
        model_test = multi_rate(num_classes=args.n_class, first_conv=args.n_feature, conv_kernels=16, backbone=False)
    else:
        NotImplementedError

    model_test.load_state_dict(best_model)
    model_test = model_test.to(DEVICE)
    acc, mf1, mr = test(args, test_loader, model_test, DEVICE, criterion, plt=False)
    return acc, mf1, mr


# Domains for each dataset
def set_domain(args):
    if args.dataset == 'dalia':
        if args.data_type == 'ecg':
            args.out_dim = 640
            args.fs = 80
            return [i for i in range(0, 15)]
        elif args.data_type == 'ppg': # 8 seconds of PPG data
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
        return [i for i in range(0, 15)]
    elif args.dataset == 'ieee_small': # 8 seconds of PPG data
        args.out_dim = 200
        args.fs, args.lowest = 25, 30
        args.data_type = 'ppg'
        return [i for i in range(0, 12)]
    elif args.dataset == 'ieee_big':
        args.out_dim = 200
        args.fs = 25
        args.data_type = 'ppg'
        return [i for i in range(0, 22)][-5:]
    elif args.dataset == 'bidmc':
        args.out_dim = 4000
        args.fs = 125
        args.data_type = 'resp'
        return [i for i in range(0, 10)]
    elif args.dataset == 'capno' or args.dataset == 'capno_64':
        args.out_dim = 800 if args.dataset == 'capno' else 1600
        args.fs, args.lowest = 25, 4
        args.data_type = 'resp'
        return [i for i in range(0, 10)]
    elif args.dataset == 'clemson':
        args.out_dim = 480
        args.fs, args.lowest = 15, 29
        args.data_type = 'step'
        return [i for i in range(0, 10)]

# python main_supervised_baseline.py --dataset 'ieee_small' --data_type 'ppg' --cuda 1

if __name__ == '__main__':
    set_seed(40)
    args = parser.parse_args()
    domain = set_domain(args)
    seed_errors = []
    for i in range(3):
        all_metrics = []
        for k in domain:
            setattr(args, 'target_domain', int(k))
            setattr(args, 'save', args.dataset + str(k))
            setattr(args, 'cases', 'subject_val')
            mif,maf, mr = train_sup(args, i)
            all_metrics.append([mif,maf, mr])
        values = np.array(all_metrics)
        mean = np.mean(values,0)
        print('MSE: {}, RMSE: {}, Mean R1: {}'.format(mean[0],mean[1], mean[2]))
        seed_errors.append([mean[0],mean[1], mean[2]])
    if args.dataset == 'clemson':
        print('Mean MAPE: {}, MSE: {}'.format(np.mean(seed_errors,0)[0],np.mean(seed_errors,0)[1]))
        print('Std MAPE: {}, MSE: {}'.format(np.std(seed_errors,0)[0],np.std(seed_errors,0)[1]))
    else:
        print('Mean MSE: {}, RMSE: {}, Mean R1: {}'.format(np.mean(seed_errors,0)[0],np.mean(seed_errors,0)[1], np.mean(seed_errors,0)[2]))
        print('Std MSE: {}, RMSE: {}, Std R1: {}'.format(np.std(seed_errors,0)[0],np.std(seed_errors,0)[1], np.std(seed_errors,0)[2]))
