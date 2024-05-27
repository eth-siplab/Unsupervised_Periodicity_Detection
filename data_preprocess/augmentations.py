import numpy as np
import torch
import scipy
from scipy.interpolate import interp1d
import random

def gen_aug(sample, ssh_type, args=None):
    if ssh_type == 'na':
        return torch.from_numpy(sample)
    elif ssh_type == 'shuffle':
        return shuffle(sample)
    elif ssh_type == 'jit_scal':
        scale_sample = scaling(jitter(sample), sigma=2)
        return torch.from_numpy(scale_sample)
    elif ssh_type == 'perm_jit':
        return torch.from_numpy(jitter(permutation(sample, max_segments=5), sigma=0.5))
    elif ssh_type == 'resample':
        return torch.from_numpy(resample(sample))
    elif ssh_type == 'resample_2':
        lin_samples, lin_ratio = linear_resample(sample, args)
        return lin_samples, torch.from_numpy(lin_ratio)
    elif ssh_type == 'freq_shift':
        shifted_samples, lin_ratio = freq_shift(sample, args)
        return shifted_samples, torch.from_numpy(lin_ratio)
    elif ssh_type == 'noise':
        return torch.from_numpy(jitter(sample))
    elif ssh_type == 'scale':
        return torch.from_numpy(scaling(sample))
    elif ssh_type == 'negate':
        return torch.from_numpy(negated(sample))
    elif ssh_type == 'shift':
        return shift_random(sample)
    elif ssh_type == 't_flip':
        return torch.from_numpy(time_flipped(sample).copy())
    elif ssh_type == 'rotation':
        if isinstance(multi_rotation(sample), np.ndarray):
            return torch.from_numpy(multi_rotation(sample))
        else:
            return multi_rotation(sample)
    elif ssh_type == 'perm':
        return torch.from_numpy(permutation(sample, max_segments=5))
    elif ssh_type == 't_warp':
        return torch.from_numpy(time_warp(sample))
    elif ssh_type == 'random_out':
        return torch.from_numpy(aug_random_zero_out(sample))
    else:
        print('The task is not available!\n')



def aug_random_zero_out(x, max_len=0):
    N, L, _ = x.shape
    max_len = L/10
    out = x.copy()
    for i in range(N):
        # Generate random start and end points for the section to be zeroed out
        start = np.random.randint(0, L - 1)
        end = min(start + np.random.randint(1, max_len), L - 1)
        # Zero out the section
        out[i, :, start:end] = 0
    
    return out


def shuffle(x):
    sample_ssh = []
    for data in x:
        p = np.random.RandomState(seed=21).permutation(data.shape[1])
        data = data[:, p]
        sample_ssh.append(data)
    return torch.stack(sample_ssh)

def shift_random(x):
    sample_ssh = []
    for data in x:
        shift = np.random.randint(0, data.shape[0])
        data = np.roll(data, shift, axis=0)
        sample_ssh.append(data)
    return np.stack(sample_ssh)

def jitter(x, sigma=0.3):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1): # apply same distortion to the signals from each sensor
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[1]))
    ai = []
    for i in range(x.shape[2]):
        xi = x[:, :, i]
        ai.append(np.multiply(xi, factor[:, :])[:, :, np.newaxis])
    return np.concatenate((ai), axis=2)


def negated(X):
    return X * -1


def time_flipped(X):
    return np.flip(X,-1)

def soft_time_flipped(X):
    reverse_channels = torch.randperm(9)[:3]
    X[:, :, reverse_channels] = torch.flip(X[:, :, reverse_channels], dims=[1])
    return X

def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[1])
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            np.random.shuffle(splits)
            warp = np.concatenate(splits).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret


def resample(x):
    from scipy.interpolate import interp1d
    orig_steps = np.arange(x.shape[1])
    interp_steps = np.arange(0, orig_steps[-1]+0.001, 1/3)
    Interp = interp1d(orig_steps, x, axis=1)
    InterpVal = Interp(interp_steps)
    start = random.choice(orig_steps)
    resample_index = np.arange(start, 3 * x.shape[1], 2)[:x.shape[1]]
    return InterpVal[:, resample_index, :]

def freq_shift(x, args):
    from scipy.signal import hilbert
    t = np.linspace(0, x.shape[1]/args.fs, x.shape[1])
    w_0 = np.random.normal(-0.05, 0.1, size=x.shape[0]) # In Hz
    lin_ratio, lin_samples = np.zeros((x.shape[0], 1)), np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        asignal = hilbert(x[i])
        shifted_signal = asignal * np.exp(2j * np.pi * w_0[i] * t)
        lin_samples[i,:] = np.real(shifted_signal)
        lin_ratio[i, :] = w_0[i] 
    return lin_samples, lin_ratio

def linear_resample(x, args):
    from scipy.interpolate import interp1d
    original_duration = x.shape[1] / args.fs
    cut_window = np.random.uniform(low=2*original_duration/3, high=original_duration, size=x.shape[0])
    lin_ratio, lin_samples = np.zeros((x.shape[0], 1)), np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        x_shorter = x[i, :int(cut_window[i] * args.fs)]
        lin_ratio[i] = x_shorter.shape[0] / x.shape[1]
        orig_steps = np.arange(0, x_shorter.shape[0]/args.fs, 1/args.fs)
        Interp = interp1d(orig_steps, x_shorter)

        interp_steps = np.linspace(0, orig_steps[-1], x.shape[1])
        lin_samples[i,:] = Interp(interp_steps)
    return lin_samples, lin_ratio

def simper_speed_change(x, target, args):
    B, M = x.shape[0], args.view_size
    periodicity_variant, speeds_labels = torch.zeros(B, M, x.shape[1]), torch.zeros(B, M)
    for i in range(B):
        speeds = torch.FloatTensor(M).uniform_(0.5, 1.5) if i < B - 2 else torch.FloatTensor(M).uniform_(0.5, 1)
        resampled = simper_speed_change_sample(x, i, target, speeds, args)
        periodicity_variant[i, :, :] = torch.from_numpy(resampled)
        speeds_labels[i,:] = speeds
    decimated_tensor = periodicity_variant[:, :, 0::args.downsample_ratio]
    # Calculate mean and standard deviation along the time dimension
    mean = decimated_tensor.mean(dim=2, keepdim=True)
    std = decimated_tensor.std(dim=2, unbiased=False, keepdim=True)
    znormed = (decimated_tensor - mean) / std
    return znormed.detach().cpu().numpy(), speeds_labels


def simper_speed_change_sample(x, index, target, speed, args):
    current_segment = x[index,:].squeeze()
    resampled = np.zeros((speed.shape[0], current_segment.shape[0]))
    for idx, k in enumerate(speed):
        x1 = simper_concat_seq(x, index, target, k, args) if k > 1 else current_segment[0:int(k*current_segment.shape[0])] 
        orig_steps = np.arange(0, x1.shape[0], 1)
        Interp = interp1d(orig_steps, x1)
        interp_steps = np.linspace(0, orig_steps[-1], current_segment.shape[0])
        resampled[idx,:] = Interp(interp_steps)
    return resampled


def simper_concat_seq(x, index, target, speed, args):
    if target[index+1] - target[index] < 10:
        overlap_length = args.fs * args.overlap * args.downsample_ratio
        continous_segment = np.concatenate((x[index, :], x[index+1, :overlap_length+1], x[index+2, :overlap_length*2+2])).squeeze()
        return continous_segment[0:int(speed*x.shape[1])] 
    else: 
        overlap_length = args.fs * args.overlap * args.downsample_ratio
        continous_segment = np.concatenate((x[index, :], x[index+1, :overlap_length+1], x[index+2, :overlap_length*2+2])).squeeze()
        return continous_segment[0:int(speed*x.shape[1])] 
        import pdb;pdb.set_trace();
        return x[index,0:int((speed/2)*x.shape[0])] 

def multi_rotation(x):
    n_channel = x.shape[2]
    n_rot = n_channel // 3
    x_rot = np.array([])
    for i in range(n_rot):
        x_rot = np.concatenate((x_rot, rotation(x[:, :, i * 3:i * 3 + 3])), axis=2) if x_rot.size else rotation(
            x[:, :, i * 3:i * 3 + 3])
    return x_rot

def rotation(X):
    """
    Applying a random 3D rotation
    """
    axes = np.random.uniform(low=-1, high=1, size=(X.shape[0], X.shape[2]))
    angles = np.random.uniform(low=-np.pi, high=np.pi, size=(X.shape[0]))
    matrices = axis_angle_to_rotation_matrix_3d_vectorized(axes, angles)
    return np.matmul(X, matrices)

def axis_angle_to_rotation_matrix_3d_vectorized(axes, angles):
    """
    Get the rotational matrix corresponding to a rotation of (angle) radian around the axes
    Reference: the Transforms3d package - transforms3d.axangles.axangle2mat
    Formula: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    """
    axes = axes / np.linalg.norm(axes, ord=2, axis=1, keepdims=True)
    x = axes[:, 0]; y = axes[:, 1]; z = axes[:, 2]
    c = np.cos(angles)
    s = np.sin(angles)
    C = 1 - c

    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC

    m = np.array([
        [ x*xC+c,   xyC-zs,   zxC+ys ],
        [ xyC+zs,   y*yC+c,   yzC-xs ],
        [ zxC-ys,   yzC+xs,   z*zC+c ]])
    matrix_transposed = np.transpose(m, axes=(2,0,1))
    return matrix_transposed

def get_cubic_spline_interpolation(x_eval, x_data, y_data):
    """
    Get values for the cubic spline interpolation
    """
    cubic_spline = scipy.interpolate.CubicSpline(x_data, y_data)
    return cubic_spline(x_eval)


def time_warp(X, sigma=0.2, num_knots=4):
    """
    Stretching and warping the time-series
    """
    time_stamps = np.arange(X.shape[1])
    knot_xs = np.arange(0, num_knots + 2, dtype=float) * (X.shape[1] - 1) / (num_knots + 1)
    spline_ys = np.random.normal(loc=1.0, scale=sigma, size=(X.shape[0] * X.shape[2], num_knots + 2))

    spline_values = np.array([get_cubic_spline_interpolation(time_stamps, knot_xs, spline_ys_individual) for spline_ys_individual in spline_ys])

    cumulative_sum = np.cumsum(spline_values, axis=1)
    distorted_time_stamps_all = cumulative_sum / cumulative_sum[:, -1][:, np.newaxis] * (X.shape[1] - 1)

    X_transformed = np.empty(shape=X.shape)
    for i, distorted_time_stamps in enumerate(distorted_time_stamps_all):
        X_transformed[i // X.shape[2], :, i % X.shape[2]] = np.interp(time_stamps, distorted_time_stamps, X[i // X.shape[2], :, i % X.shape[2]])
    return X_transformed

