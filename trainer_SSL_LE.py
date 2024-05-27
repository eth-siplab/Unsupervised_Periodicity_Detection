import torch
import torch.nn as nn
import numpy as np
import os
import pickle as cp
from data_preprocess.augmentations import gen_aug, simper_speed_change
from new_augmentations import *
from models.frameworks import *
from models.loss import *
from models.backbones import *
from models.models_nc import ResNet1D
from plot_latent_vs_true import * 
from sklearn.metrics import roc_auc_score
from data_preprocess import data_preprocess_IEEE_small
from data_preprocess import data_preprocess_IEEE_big
from data_preprocess import data_preprocess_dalia
from data_preprocess import data_preprocess_ptb
from data_preprocess import data_preprocess_wesad
from data_preprocess import data_preprocess_clemson
from data_preprocess import data_preprocess_capno
from data_preprocess import data_preprocess_bidmc

from sklearn.metrics import f1_score
from scipy.special import softmax
import seaborn as sns
import fitlog
from copy import deepcopy

# create directory for saving models and plots
global model_dir_name
model_dir_name = 'results'
if not os.path.exists(model_dir_name):
    os.makedirs(model_dir_name)
global plot_dir_name
plot_dir_name = 'plot'
if not os.path.exists(plot_dir_name):
    os.makedirs(plot_dir_name)


def setup_dataloaders(args):
    if args.dataset == 'ieee_small':
        args.n_feature = 1
        args.len_sw = 200
        args.n_class = 180 # 30 -- 210 bpm
        train_loaders, val_loader, test_loader = data_preprocess_IEEE_small.prep_ieee_small(args)
    if args.dataset == 'ieee_big':
        args.n_feature = 1
        args.len_sw = 200
        args.n_class = 180 # 30 -- 210 bpm
        train_loaders, val_loader, test_loader = data_preprocess_IEEE_big.prep_ieeebig(args)     
    if args.dataset == 'dalia':
        args.n_feature = 1
        args.len_sw = 200
        args.n_class = 180 # 30 -- 210 bpm
        train_loaders, val_loader, test_loader = data_preprocess_dalia.prep_dalia(args)              
    if args.dataset == 'ptb':
        args.n_feature = 1  # 1 channel
        args.len_sw = args.out_dim  # length of the signal
        args.n_class = 180 # 30 -- 210 bpm
        train_loaders, val_loader, test_loader = data_preprocess_ptb.prep_ptb(args)           
    if args.dataset == 'wesad':
        args.n_feature = 1 # 1 channel
        args.len_sw = args.out_dim # length of the signal
        args.n_class = 200 # 30 -- 230 bpm
        train_loaders, val_loader, test_loader = data_preprocess_wesad.prep_wesad(args) 
    if args.dataset == 'capno' or args.dataset == 'capno_64':
        args.n_feature = 1
        args.len_sw = args.out_dim
        args.n_class = 40 # 3 -- 43 rpm
        train_loaders, val_loader, test_loader = data_preprocess_capno.prep_capno(args) 
    if args.dataset == 'bidmc':
        args.n_feature = 1
        args.len_sw = args.out_dim
        args.n_class = 22 # 5 -- 27 rpm
        train_loaders, val_loader, test_loader = data_preprocess_bidmc.prep_bidmc(args) 
    if args.dataset == 'clemson_semi':
        args.n_feature = 1 # 1 channel
        args.len_sw = args.out_dim # length of the signal
        args.n_class = 48 # 20 -- 67 count
        train_loaders, val_loader, test_loader = data_preprocess_clemson.prep_clemson(args)         
    if args.dataset == 'clemson':
        args.n_feature = 1
        args.len_sw = args.out_dim
        args.n_class = 49
        train_loaders, val_loader, test_loader = data_preprocess_clemson.prep_clemson(args)
    return train_loaders, val_loader, test_loader


def setup_linclf(args, DEVICE, bb_dim):
    '''
    @param bb_dim: output dimension of the backbone network
    @return: a linear classifier
    '''
    classifier = Classifier(bb_dim=bb_dim, n_classes=args.n_class)
    classifier.classifier.weight.data.normal_(mean=0.0, std=0.01)
    classifier.classifier.bias.data.zero_()
    classifier = classifier.to(DEVICE)
    return classifier


def setup_model_optm(args, DEVICE, classifier=True):
    # set up backbone network
    if args.backbone == 'FCN':
        backbone = FCN(in_dim= args.out_dim, n_channels=args.n_feature, n_classes=args.n_class, backbone=True)
    elif args.backbone == 'FCN2':
        backbone = FCN_2(in_dim= args.out_dim, n_channels=args.n_feature, n_classes=args.n_class, backbone=True)        
    elif args.backbone == 'DCL':
        backbone = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=True)
    elif args.backbone == 'DCL2':
        backbone = DeepConvLSTM_2(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=7, LSTM_units=128, backbone=True)
    elif args.backbone == 'LSTM':
        backbone = LSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=128, backbone=True)
    elif args.backbone == 'AE':
        backbone = AE(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, outdim=128, backbone=True)
    elif args.backbone == 'CNN_AE':
        backbone = CNN_AE(n_channels=args.n_feature, n_classes=args.n_class, out_channels=128, backbone=True)
    elif args.backbone == 'Transformer':
        backbone = Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=True)
    elif args.backbone == 'UNET':
        backbone = UNET_1D_simp_ssl(input_dim=1, output_dim=args.out_dim, layer_n=32, kernel_size=5, depth=1, args=args, backbone=True)
    elif args.backbone == 'RESNET':
        backbone = ResNet1D(in_channels=1, base_filters=32, kernel_size=5, stride=1, groups=1, n_block=3, n_classes=args.n_class, downsample_gap=2, increasefilter_gap=4, output_dim=args.out_dim, backbone=True)
    else:
        NotImplementedError

    # set up model and optimizers
    if args.framework in ['byol', 'simsiam']:
        model = BYOL(DEVICE, backbone, window_size=args.len_sw, n_channels=args.n_feature, projection_size=args.p,
                     projection_hidden_size=args.phid, moving_average=args.EMA)
        optimizer1 = torch.optim.Adam(model.online_encoder.parameters(),
                                      args.lr,
                                      weight_decay=args.weight_decay)
        optimizer2 = torch.optim.Adam(model.online_predictor.parameters(),
                                      args.lr * args.lr_mul,
                                      weight_decay=args.weight_decay)
        optimizers = [optimizer1, optimizer2]
    elif args.framework == 'simclr' or args.framework == 'vicreg' or args.framework == 'barlowtwins': # Same models, different losses
        model = SimCLR(backbone=backbone, dim=args.p)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizers = [optimizer]
    elif args.framework == 'nnclr':
        model = NNCLR(backbone=backbone, dim=args.p, pred_dim=args.phid)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizers = [optimizer]
    elif args.framework == 'tstcc':
        model = TSTCC(backbone=backbone, DEVICE=DEVICE, temp_unit=args.temp_unit, tc_hidden=100)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        optimizers = [optimizer]
    elif args.framework == 'simper':
        model = SimPer(backbone=backbone, dim=args.p)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        optimizers = [optimizer]
    elif args.framework == 'ts2vec': # dummy models for ts2vec
        model = SimCLR(backbone=backbone, dim=args.p)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizers = [optimizer]
    else:
        NotImplementedError

    model = model.to(DEVICE)

    # set up linear classfier
    if classifier:
        bb_dim = backbone.out_dim
        classifier = setup_linclf(args, DEVICE, bb_dim)
        return model, classifier, optimizers

    else:
        return model, optimizers


def delete_files(args):
    for epoch in range(args.n_epoch):
        model_dir = model_dir_name + '/pretrain_' + args.model_name + str(epoch) + '.pt'
        if os.path.isfile(model_dir):
            os.remove(model_dir)

        cls_dir = model_dir_name + '/lincls_' + args.model_name + str(epoch) + '.pt'
        if os.path.isfile(cls_dir):
            os.remove(cls_dir)


def setup(args, DEVICE):
    # set up default hyper-parameters
    if args.framework == 'byol':
        args.weight_decay = 1.5e-6
    if args.framework == 'simsiam':
        args.weight_decay = 1e-4
        args.EMA = 0.0
        args.lr_mul = 1.0
    if args.framework in ['simclr', 'nnclr']:
        args.criterion = 'NTXent'
        args.weight_decay = 1e-6
    if args.framework == 'tstcc':
        args.criterion = 'NTXent'
        args.backbone = 'FCN'
        args.weight_decay = 3e-4
    if args.framework == 'simper':
        args.criterion = 'Cont_InfoNCE'
        args.backbone = 'UNET' # Cares about FFT
        args.weight_decay = 1e-6
    if args.framework == 'vicreg':
        args.criterion = 'VICReg'
        args.weight_decay = 1e-6
    if args.framework == 'barlowtwins':
        args.criterion = 'barlowtwins'
        args.weight_decay = 1.5e-6

    model, classifier, optimizers = setup_model_optm(args, DEVICE, classifier=True)

    # loss fn
    if args.criterion == 'cos_sim':
        criterion = nn.CosineSimilarity(dim=1)
    elif args.criterion == 'NTXent':
        if args.framework == 'tstcc':
            criterion = NTXentLoss(DEVICE, args.batch_size, temperature=0.2)
        else:
            criterion = NTXentLoss(DEVICE, args.batch_size, temperature=0.1)
    elif args.criterion == 'Cont_InfoNCE':
        criterion = Cont_InfoNCE(DEVICE, args.batch_size, temperature=0.1)
    elif args.criterion == 'VICReg':
        criterion = VICReg(args)
    elif args.criterion == 'barlowtwins':
        criterion = BarlowTwins(args)

    args.model_name = 'try_scheduler_' + args.framework + '_pretrain_' + args.dataset + '_eps' + str(args.n_epoch) + '_lr' + str(args.lr) + '_bs' + str(args.batch_size) \
                      + '_aug1' + args.aug1 + '_aug2' + args.aug2 + '_dim-pdim' + str(args.p) + '-' + str(args.phid) \
                      + '_EMA' + str(args.EMA) + '_criterion_' + args.criterion + '_lambda1_' + str(args.lambda1) + '_lambda2_' + str(args.lambda2) + '_tempunit_' + args.temp_unit

    criterion_cls = nn.CrossEntropyLoss() 
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=args.lr_cls)

    schedulers = []
    for optimizer in optimizers:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch, eta_min=0)
        schedulers.append(scheduler)

    global nn_replacer
    nn_replacer = None
    if args.framework == 'nnclr':
        nn_replacer = NNMemoryBankModule(size=args.mmb_size)

    global recon
    recon = None
    if args.backbone in ['AE', 'CNN_AE']:
        recon = nn.MSELoss()

    return model, optimizers, schedulers, criterion, classifier, criterion_cls, optimizer_cls

def calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=None, nn_replacer=None, view_learner=None):
    sample = sample.transpose(2,1) # sample --> (Batch_size, time steps, channel size)
    sample = sample.detach().cpu().numpy()
    if args.framework == 'simper':
        samples, speed_rate = simper_speed_change(sample, target, args)  # Obtain the M variant views for samples --> [bsz, M, W, C] // Periodicity-Variant views
        aug_sample1, aug_sample2 = gen_aug(samples, 'noise'), gen_aug(samples, 'noise') # Shape --> [bsz, 2*M, W, C] // Periodicity-Invariant views
        speed_rate = speed_rate.to(DEVICE)
    else:
        aug_sample1, aug_sample2 = gen_aug(sample, args.aug1), gen_aug(sample, args.aug2) # Shape --> (Batch_size, number of inputs, channel size)

    if args.Randomfftmix:
        aug_sample1, aug_sample2 = gen_aug(sample, args.aug1), gen_new_aug(gen_aug(sample, args.aug2), args, DEVICE)

    aug_sample1, aug_sample2, target = aug_sample1.to(DEVICE).float(), aug_sample2.to(DEVICE).float(), target.to(DEVICE).long()
    if args.framework in ['byol', 'simsiam']:
        assert args.criterion == 'cos_sim'
    if args.framework in ['tstcc', 'simclr', 'nnclr']:
        assert args.criterion == 'NTXent'
    if args.framework in ['byol', 'simsiam', 'nnclr']:
        if args.backbone in ['AE', 'CNN_AE']:
            x1_encoded, x2_encoded, p1, p2, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
            recon_loss = recon(aug_sample1, x1_encoded) + recon(aug_sample2, x2_encoded)
        else:
            p1, p2, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
        if args.framework == 'nnclr':
            z1 = nn_replacer(z1, update=False)
            z2 = nn_replacer(z2, update=True)
        if args.criterion == 'cos_sim':
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        elif args.criterion == 'NTXent':
            loss = (criterion(p1, z2) + criterion(p2, z1)) * 0.5
        if args.backbone in ['AE', 'CNN_AE']:
            loss = loss * args.lambda1 + recon_loss * args.lambda2
    if args.framework == 'simclr' or args.framework == 'vicreg' or args.framework == 'barlowtwins':
        if args.backbone in ['AE', 'CNN_AE']:
            x1_encoded, x2_encoded, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
            recon_loss = recon(aug_sample1, x1_encoded) + recon(aug_sample2, x2_encoded)
            loss = loss * args.lambda1 + recon_loss * args.lambda2
        else:
            z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
            loss = criterion(z1, z2) 
    if args.framework == 'tstcc':
        nce1, nce2, p1, p2 = model(x1=aug_sample1, x2=aug_sample2)
        tmp_loss = nce1 + nce2
        ctx_loss = criterion(p1, p2)
        loss = tmp_loss * args.lambda1 + ctx_loss * args.lambda2
    if args.framework == 'simper':
        aug_sample1_model = aug_sample1.reshape(aug_sample1.shape[0] * aug_sample1.shape[1], aug_sample1.shape[2], 1)
        aug_sample2_model = aug_sample2.reshape(aug_sample2.shape[0] * aug_sample2.shape[1], aug_sample2.shape[2], 1)
        # import pdb;pdb.set_trace();
        z1, z2 = model(x1=aug_sample1_model, x2=aug_sample2_model) # Feed all samples to model for efficiency
        z1 = z1.reshape(aug_sample1.shape[0], aug_sample1.shape[1], -1) # Reshape back to [bsz, M, Time steps]
        z2 = z2.reshape(aug_sample2.shape[0], aug_sample2.shape[1], -1)
        speed_rate = label_distance(speed_rate, speed_rate)
        for i in range(z1.shape[0]):
            if i == 0:
                loss = criterion(z1[i, :, :], z2[i, :, :], speed_rate[i,:])
            else:
                loss += criterion(z1[i, :, :], z2[i, :, :], speed_rate[i,:])
        loss /= z1.shape[0]
    return loss


def train(train_loaders, val_loader, model, DEVICE, optimizers, schedulers, criterion, args):
    best_model = None
    min_val_loss = 1e8
    for epoch in range(args.n_epoch):
        #logger.debug(f'\nEpoch : {epoch}')
        total_loss = 0
        n_batches = 0
        model.train()
        for idx, train_x in enumerate(train_loaders):
            sample, target = train_x[0], train_x[1]
            for optimizer in optimizers:
                optimizer.zero_grad()
            if sample.size(0) != args.batch_size:
                continue
            n_batches += 1

            loss = calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=recon, nn_replacer=nn_replacer)

            total_loss += loss.item()
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            if args.framework in ['byol', 'simsiam']:
                model.update_moving_average()                 

        for scheduler in schedulers:
            scheduler.step()

        # save model
        model_dir = model_dir_name + '/pretrain_' + args.model_name + str(epoch) + '.pt'
        #print('Saving model at {} epoch to {}'.format(epoch, model_dir))
        torch.save({'model_state_dict': model.state_dict()}, model_dir)

        if args.cases in ['subject', 'subject_large', 'subject_large_ssl_fn']:
            with torch.no_grad():
                best_model = copy.deepcopy(model.state_dict())
        else:
            with torch.no_grad():
                model.eval()
                total_loss = 0
                n_batches = 0
                for idx, (sample, target, domain) in enumerate(val_loader):
                    if sample.size(0) != args.batch_size:
                        continue
                    n_batches += 1
                    loss = calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=recon, nn_replacer=nn_replacer)
                    total_loss += loss.item()
                if total_loss <= min_val_loss:
                    min_val_loss = total_loss
                    best_model = copy.deepcopy(model.state_dict())
                # logger.debug(f'Val Loss     : {total_loss / n_batches:.4f}')
                # fitlog.add_loss(total_loss / n_batches, name="pretrain validation loss", step=epoch)
    return best_model


def test(test_loader, best_model, DEVICE, criterion, args): # Test the pre-trained model --> to observe the features
    model, _ = setup_model_optm(args, DEVICE, classifier=False)
    model.load_state_dict(best_model)

    return model


def lock_backbone(model, args):
    if args.framework not in ['ts2vec']:
        for name, param in model.named_parameters():
            param.requires_grad = False
    else:
        for name, param in model._net.named_parameters():
            param.requires_grad = False

    if args.framework in ['simsiam', 'byol']:
        trained_backbone = model.online_encoder.net
    elif args.framework in ['simclr', 'simper', 'nnclr', 'tstcc','vicreg', 'barlowtwins']:
        trained_backbone = model.encoder
    elif args.framework in ['ts2vec']:
        trained_backbone = model
    else:
        NotImplementedError

    return trained_backbone


def calculate_lincls_output(sample, target, trained_backbone, classifier, criterion, args):
    lowest = args.lowest
    sample = sample.transpose(2,1) # sample --> (Batch_size, time steps, channel size)
    target = target.round().long() - lowest  

    target = torch.clamp(target, min=0)
    #if (target < 0).any() or (target > 180).any(): import pdb;pdb.set_trace();
    if args.framework not in ['ts2vec']:
        _, feat = trained_backbone(sample)
    else:
        feat = torch.from_numpy(trained_backbone.encode(sample.detach().cpu().numpy(), encoding_window='full_series')).to(args.cuda)

    if len(feat.shape) == 3:
        feat = feat.reshape(feat.shape[0], -1)

    output = classifier(feat).squeeze()

    loss = criterion(output, target)
    _, predicted = torch.max(output.data, 1)
    return loss, predicted, feat


def train_lincls(train_loaders, val_loader, trained_backbone, classifier, DEVICE, optimizer, criterion, args):
    best_lincls = None
    min_val_loss = 1e8
    
    # if args.plot_tsne:
    #     import pdb;pdb.set_trace();
    #     plot_vs_gt_usc(vae, train_loaders.dataset, 'train', z_inds=None)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch, eta_min=0)

    for epoch in range(args.n_epoch):
        classifier.train()
        for idx, train_x in enumerate(train_loaders):
            sample, target = train_x[0], train_x[1]

            loss, predicted, _ = calculate_lincls_output(sample, target, trained_backbone, classifier, criterion, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # save model
        model_dir = model_dir_name + '/lincls_' + args.model_name + str(epoch) + '.pt'
        if args.framework not in ['ts2vec']:
            torch.save({'trained_backbone': trained_backbone.state_dict(), 'classifier': classifier.state_dict()}, model_dir) 

        if args.scheduler:
            scheduler.step()

        if args.cases in ['subject', 'subject_large','subject_large_ssl_fn']:
            with torch.no_grad():
                best_lincls = copy.deepcopy(classifier.state_dict())
        else:
            with torch.no_grad():
                classifier.eval()
                total_loss = 0
                total = 0
                correct = 0
                for idx, (sample, target, domain) in enumerate(val_loader):
                    sample, target = sample.to(DEVICE).float(), target.to(DEVICE)
                    loss, predicted, _ = calculate_lincls_output(sample, target, trained_backbone, classifier, criterion, args)
                    total_loss += loss.item()
                    total += target.size(0)
                    correct += (predicted == target).sum()
                acc_val = float(correct) * 100.0 / total
                if total_loss <= min_val_loss:
                    min_val_loss = total_loss
                    best_lincls = copy.deepcopy(classifier.state_dict())
    return best_lincls


def test_lincls(test_loader, trained_backbone, best_lincls, DEVICE, criterion, args, plt=False):  # Test the fine-tuned model
    classifier = setup_linclf(args, DEVICE, trained_backbone.out_dim) if args.framework not in ['ts2vec'] else setup_linclf(args, DEVICE, args.p)
    classifier.load_state_dict(best_lincls)
    total_loss = 0
    feats = None
    trgs = np.array([])
    preds = np.array([])
    otp = np.array([])
    with torch.no_grad():
        classifier.eval()
        for idx, testx in enumerate(test_loader):
            sample, target = testx[0], testx[1]
            loss, predicted, feat = calculate_lincls_output(sample, target, trained_backbone, classifier, criterion, args)
            total_loss += loss.item()
            if feats is None:
                feats = feat
            else:
                feats = torch.cat((feats, feat), 0)
            trgs = np.append(trgs, target.data.cpu().numpy())
            preds = np.append(preds, predicted.data.cpu().numpy() +  args.lowest) # go back to bpm from bins
        if args.data_type == 'step':
            trgs, preds = trgs + args.lowest, preds + args.lowest
            print(f'MAPE: {100*np.mean(np.abs((trgs-preds)/trgs))}, MAE: {np.mean(np.abs(preds-trgs))}')
        else:
            print(f'MSE: {np.mean(np.abs(preds-trgs))}, RMSE: {np.sqrt(np.mean(np.square(preds-trgs)))}, r2: {np.corrcoef(preds, trgs)[0,1]}')
    
    if plt == True:
        tsne(feats, trgs, save_dir=plot_dir_name + '/' + args.model_name + '_tsne.png')
        mds(feats, trgs, save_dir=plot_dir_name + '/' + args.model_name + '_mds.png')
        print('plots saved to ', plot_dir_name)
    if args.data_type == 'step':
        return np.array([100*np.mean(np.abs((trgs-preds)/trgs)), np.mean(np.abs(preds-trgs)), 1])
    else:
        return np.array([np.mean(np.abs(preds-trgs)), np.sqrt(np.mean(np.square(preds-trgs))), np.corrcoef(preds, trgs)[0,1]])