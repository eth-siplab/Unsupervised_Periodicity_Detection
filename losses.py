import torch

EPSILON = 1e-10
BP_LOW=30
BP_HIGH=210
BP_DELTA=6


"""
https://github.com/CVRL/SiNC-rPPG/blob/main/src/utils/losses.py

"""

def _IPR_SSL(freqs, psd, low_hz=BP_LOW, high_hz=BP_HIGH, device=None):
    use_freqs = torch.logical_and(freqs >= low_hz, freqs <= high_hz)
    zero_freqs = torch.logical_not(use_freqs)
    use_energy = torch.sum(psd[:,use_freqs], dim=1)
    zero_energy = torch.sum(psd[:,zero_freqs], dim=1)
    denom = use_energy + zero_energy + EPSILON
    ipr_loss = torch.sum(zero_energy / denom)
    return ipr_loss


def _SNR_SSL(freqs, psd, low_hz=BP_LOW, high_hz=BP_HIGH, freq_delta=BP_DELTA, normalized=False, bandpassed=False, device=None):
    if not bandpassed:
        freqs, psd = ideal_bandpass(freqs, psd, low_hz, high_hz)
    signal_freq_idx = torch.argmax(psd, dim=1)
    signal_freq = freqs[signal_freq_idx].view(-1,1)
    freqs = freqs.repeat(psd.shape[0],1)

    low_cut = signal_freq - freq_delta
    high_cut = signal_freq + freq_delta
    band_idcs = torch.logical_and(freqs >= low_cut, freqs <= high_cut).to(device)
    signal_band = torch.sum(psd * band_idcs, dim=1)
    noise_band = torch.sum(psd * torch.logical_not(band_idcs), dim=1)
    denom = signal_band + noise_band + EPSILON
    snr_loss = torch.sum(noise_band / denom)
    return snr_loss


def _EMD_SSL(freqs, psd, low_hz=BP_LOW, high_hz=BP_HIGH, normalized=False, bandpassed=False, device=None):
    ''' Squared earth mover's distance to uniform distribution.
    '''
    if not bandpassed:
        freqs, psd = ideal_bandpass(freqs, psd, low_hz, high_hz)
    if not normalized:
        psd = normalize_psd(psd)

    B,T = psd.shape
    psd = torch.sum(psd, dim=0) / B
    expected = ((1/T)*torch.ones(T)).to(device) #uniform distribution
    emd_loss = torch.sum(torch.square(torch.cumsum(psd, dim=0) - torch.cumsum(expected, dim=0)))
    return emd_loss



def ideal_bandpass(freqs, psd, low_hz, high_hz):
    freq_idcs = torch.logical_and(freqs >= low_hz, freqs <= high_hz)
    freqs = freqs[freq_idcs]
    psd = psd[:,freq_idcs]
    return freqs, psd


def normalize_psd(psd):
    return psd / torch.sum(psd, keepdim=True, dim=1) ## treat as probabilities




################################### when the resample augmentation is used ###################################

def resample_v2_loss(x_fft, org_fft, lin_ratio, loc_0, loc_1, freq, args):
    l1_loss, l2_loss = torch.ones((x_fft.size(0),1)).cuda(args.cuda), torch.ones((x_fft.size(0),1)).cuda(args.cuda)
    l3_loss, bpms = torch.ones((x_fft.size(0),1)).cuda(args.cuda), torch.ones((x_fft.size(0),1)).cuda(args.cuda)
    for i in range(len(x_fft)):
        loc0, loc1 = (loc_0*lin_ratio[i]).int(), (loc_1*lin_ratio[i]).int()
        l1_loss[i] = (torch.sum(x_fft[i,:,0:loc0], dim=1).squeeze() + torch.sum(x_fft[i,:,loc1:], dim=1).squeeze())
        freq_interest = x_fft[i,:,loc0:loc1]/torch.sum(x_fft[i,:,loc0:loc1], axis=1, keepdim=True)
        freq_interest_org = org_fft[i,:,loc0:loc1]/torch.sum(org_fft[i,:,loc0:loc1], axis=1, keepdim=True)

        l2_loss[i] = -torch.sum(freq_interest*torch.log(freq_interest), dim=1)
        l3_loss[i] = 3*torch.nn.KLDivLoss(reduction='sum')(torch.log(freq_interest), freq_interest_org)

        peak_locs = torch.argmax(x_fft[i, 0, loc0:loc1]).cpu()
        bpms[i] = freq[peak_locs+loc0]

    return torch.sum(l1_loss+l2_loss+l3_loss), bpms    

def shiftfreq_loss(x_fft, org_fft, lin_ratio, loc_0, loc_1, freq, args):
    l1_loss, l2_loss = torch.ones((x_fft.size(0),1)).cuda(args.cuda), torch.ones((x_fft.size(0),1)).cuda(args.cuda)
    l3_loss, bpms = torch.ones((x_fft.size(0),1)).cuda(args.cuda), torch.ones((x_fft.size(0),1)).cuda(args.cuda)
    for i in range(len(x_fft)):
        import pdb;pdb.set_trace();
        loc0, loc1 = (loc_0*lin_ratio[i]).int(), (loc_1*lin_ratio[i]).int()
        l1_loss[i] = (torch.sum(x_fft[i,:,0:loc0], dim=1).squeeze() + torch.sum(x_fft[i,:,loc1:], dim=1).squeeze())
        freq_interest = x_fft[i,:,loc0:loc1]/torch.sum(x_fft[i,:,loc0:loc1], axis=1, keepdim=True)
        freq_interest_org = org_fft[i,:,loc0:loc1]/torch.sum(org_fft[i,:,loc0:loc1], axis=1, keepdim=True)

        l2_loss[i] = -torch.sum(freq_interest*torch.log(freq_interest), dim=1)
        l3_loss[i] = 3*torch.nn.KLDivLoss(reduction='sum')(torch.log(freq_interest), freq_interest_org)

        peak_locs = torch.argmax(x_fft[i, 0, loc0:loc1]).cpu()
        bpms[i] = freq[peak_locs+loc0]

    return torch.sum(l1_loss+l2_loss+l3_loss), bpms    
