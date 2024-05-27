import torch
from models.models_nc import setup_model
from trainer import Trainer
import argparse
from data_preprocess import data_preprocess_dalia
import pywt
import ptwt
import numpy as np
from scipy.io import savemat

def get_output(args, test_data, model, seed, target_domain):
    PATH = 'saved_models' + str(args.cuda) + '/' + args.wavelet + str(args.level) + str(seed) +'.pt'
    model.load_state_dict(torch.load(PATH))
    wavelet = pywt.Wavelet(args.wavelet)
    get_output_local(PATH, model, test_data, wavelet, seed, target_domain)
    import pdb;pdb.set_trace

def get_output_local(PATH, model, test_loader, wavelet, seed, target_domain):
    model.load_state_dict(torch.load(PATH))
    save_dic, wavelets = {}, []
    for i, test_x in enumerate(test_loader):
        y_pred, y_coeffs = model(test_x)
        time_modified = ptwt.waverec(y_pred, wavelet).squeeze()
        original = test_x.pop()
        wavelets = [arr.detach().cpu().numpy().astype(object) for arr in test_x]
        time_original = ptwt.waverec(original, wavelet).squeeze()
    time_original, time_modified = time_original.detach().cpu().numpy(), time_modified.detach().cpu().numpy()
    save_dic = {'original_data': time_original, 'output': time_modified, 'wavelets0':wavelets[0]
                ,'wavelets1':wavelets[1], 'wavelets2':wavelets[2], 'wavelets3':wavelets[3], 'wavelets4':wavelets[4]}
    save_path = 'saved_outputs' + '/' + str(target_domain) + str(seed) +'.mat'
    savemat(save_path, save_dic)
    return 

def get_output_local_sine_gen(PATH, model, test_loader, wavelet, seed, target_domain):
    model.load_state_dict(torch.load(PATH))
    save_dic, time_modified, time_original = {}, [], []
    trainer = Trainer(args, model)
    for i, test_x in enumerate(test_loader):
        y_pred, out_freq, out_phase = model(test_x)
        generated_signal,t = trainer.generate_sinusoidal(out_phase, out_freq)
        time_original.append(y_pred.squeeze().detach().cpu().numpy()), 
        time_modified.append(generated_signal.detach().cpu().numpy())
    save_dic = {'original_data': time_original, 'output': time_modified}
    save_path = 'saved_outputs' + '/' + 'sinu' + str(target_domain) + str(seed) +'.mat'
    savemat(save_path, save_dic)
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argument setting of network')
    parser.add_argument('--cuda', default=3, type=int, help='cuda device ID: 0,1,2,3')
    parser.add_argument('--wavelet', default='db6', type=str, help='wavelet type|haar,db')
    parser.add_argument('--level', default=4, type=int, help='wavelet level')
    parser.add_argument('--dataset', default='dalia', type=str, help='dataset type')
    parser.add_argument('--input_dim', default=800, type=int, help='Input size of the original signal')
    parser.add_argument('--cases', type=str, default='subject', choices=['random', 'subject'], help='name of scenarios, cross user')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size')
    parser.add_argument('--wandb', action='store_true', help='Saving')
    parser.add_argument('--target_domain', default=0, type=int, help='Target subject')
    parser.add_argument('--test', action='store_true', help='test data')
 
    args = parser.parse_args()
    DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    seed = 40
    PATH = 'saved_models' + str(args.cuda) + '/' + args.wavelet + str(args.level) + str(seed) +'.pt'
    model = setup_model(args, DEVICE)
    _, _, test_loader = data_preprocess_dalia.prep_dalia(args)
    wavelet = pywt.Wavelet(args.wavelet)
    #get_output_local(PATH, model, test_loader, wavelet, seed, args.target_domain)
    get_output_local_sine_gen(PATH, model, test_loader, wavelet, seed, args.target_domain)