import torch
import numpy as np
from torch.nn import Softmax
import torch.nn as nn

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature=0.1, use_cosine_similarity=True):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs, dim_mixing=False):
        # l2 normalized features 
        zjs = torch.nn.functional.normalize(zjs, dim=1)
        zis = torch.nn.functional.normalize(zis, dim=1)
        if dim_mixing:
            m = torch.distributions.Beta(2*torch.ones(zjs.shape[0],1), 2*torch.ones(zjs.shape[0],1))
            lamb_from_beta = (m.sample() + 1).to(zjs.device) # For positive extrapolation
            zis = torch.mul(lamb_from_beta,zis) + torch.mul((1-lamb_from_beta),zjs)
            zjs = torch.mul(lamb_from_beta,zjs) + torch.mul((1-lamb_from_beta),zis)
        
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


class Cont_InfoNCE(torch.nn.Module):

    def __init__(self, device, batch_size, temperature=0.1):
        super(Cont_InfoNCE, self).__init__()
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._max_cross_corr
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _max_cross_corr(self, feats_1, feats_2):
        # feats_1: 1 x T (# time stamp)
        # feats_2: M (# aug) x T (# time stamp)
        feats_2 = feats_2.to(feats_1.dtype)
        feats_1 = feats_1 - torch.mean(feats_1, dim=-1, keepdim=True)
        feats_2 = feats_2 - torch.mean(feats_2, dim=-1, keepdim=True)

        min_N = min(feats_1.shape[-1], feats_2.shape[-1])
        padded_N = max(feats_1.shape[-1], feats_2.shape[-1]) * 2
        feats_1_pad = torch.nn.functional.pad(feats_1, (0, padded_N - feats_1.shape[-1]))
        feats_2_pad = torch.nn.functional.pad(feats_2, (0, padded_N - feats_2.shape[-1]))
        feats_1_fft = torch.fft.rfft(feats_1_pad)
        feats_2_fft = torch.fft.rfft(feats_2_pad)
        X = feats_1_fft * torch.conj(feats_2_fft)

        power_norm = (torch.std(feats_1, dim=-1, keepdim=True) *
                    torch.std(feats_2, dim=-1, keepdim=True)).to(X.dtype)
        power_norm = torch.where(power_norm == 0, torch.ones_like(power_norm), power_norm)
        X = X / power_norm

        cc = torch.fft.irfft(X) / (min_N - 1)
        max_cc = torch.max(cc, dim=-1).values

        return max_cc

    def forward(self, zis, zjs, speeds):
        """
        zis: M (# aug) x T (# time stamp)
        zjs: M (# aug) x T (# time stamp)
        """
        # Calculate distance for a single row of x.
        def per_x_dist(i):
            return self.similarity_function(zis[i:(i + 1), :], zjs)

        # Compute and stack distances for all rows of x.
        dist = torch.stack([per_x_dist(i) for i in range(zis.shape[0])])
        loss = self.criterion(dist, speeds)

        return loss 
    



def label_distance(labels_1, labels_2, dist_fn='l1', label_temperature=0.1):
    # labels: bsz x M(#augs)
    # output: bsz x M(#augs) x M(#augs)
    if dist_fn == 'l1':
        dist_mat = - torch.abs(labels_1[:, :, None] - labels_2[:, None, :])
    elif dist_fn == 'l2':
        dist_mat = - torch.abs(labels_1[:, :, None] - labels_2[:, None, :]) ** 2
    elif dist_fn == 'sqrt':
        dist_mat = - torch.abs(labels_1[:, :, None] - labels_2[:, None, :]).sqrt()
    else:
        raise NotImplementedError(f"`{dist_fn}` not implemented.")

    prob_mat = torch.nn.functional.softmax(dist_mat / label_temperature, dim=-1)
    return prob_mat