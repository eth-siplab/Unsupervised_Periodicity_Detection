import numpy as np
from torch.utils.data import Dataset

class base_loader(Dataset):
    def __init__(self, samples, bpms, lin_ratio, args):
        self.samples = samples
        self.bpms = bpms
        self.lin_ratio = lin_ratio
        self.args = args
    
    def __getitem__(self, index):
        sample, target, lin_ratio = self.samples[index], self.bpms[index], self.lin_ratio[index]
        return sample, target, lin_ratio

    def __len__(self):
        return len(self.samples[:,0])
    

class base_loader_dalia(Dataset):
    def __init__(self, samples, bpms, lin_ratio, activities, args):
        self.samples = samples
        self.bpms = bpms
        self.args = args
        self.activities = activities
        self.lin_ratio = lin_ratio
    
    def __getitem__(self, index):
        sample, target, lin_ratio, activities = self.samples[index], self.bpms[index], self.lin_ratio, self.activities[index]
        return sample, target, lin_ratio, activities

    def __len__(self):
        return len(self.samples[:,0])
    

def get_sample_weights(y, weights):
    '''
    to assign weights to each sample
    '''
    label_unique = np.unique(y)
    sample_weights = []
    for val in y:
        idx = np.where(label_unique == val)
        sample_weights.append(weights[idx])
    return sample_weights