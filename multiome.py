import sys
import torch
import math
import numpy as np
import anndata
from tqdm import tqdm
import seq2exp_functions


class MultiOmicDataset():
    '''
    A class to organize multiome data prepared by make_pooled_datasets.
    '''
    def __init__(self, src_file, include_disp = False, include_sigma = False):

        self.__data = np.load(src_file)
        self.__n_samples = self.__data['samples'].shape[0]
        self.__seq_len = self.__data['sequence'].shape[1]

        self.__torch_promoters = None
        self.__torch_atac = None
        self.__torch_rna = None

        self.__train_indices = None
        self.__val_indices = None


    def encode_promoter(self, promoter):
        return torch.nn.functional.one_hot(torch.tensor(promoter, dtype = int), 4).T
    

    def fetch_torch_samples(self, return_samples = False, bases = 4):
        if self.__torch_promoters is None:
            promoters = self.__data['sequence']
            n = self.__n_samples
            p = torch.zeros((n, bases, self.__seq_len), dtype = int)
            for i in tqdm(range(n), total = n):
                seq = promoters[i]
                p[i] = self.encode_promoter(seq)
            a = torch.from_numpy(self.__data['atac'])
            r = torch.from_numpy(self.__data['rna'])
            self.__torch_promoters = p
            self.__torch_atac = a
            self.__torch_rna = r
        if return_samples == True:
            return self.__data['samples'], self.__torch_promoters, self.__torch_atac, self.__torch_rna
    

    def get_seq_len(self):
        return self.__seq_len


    def get_n_samples(self):
        return self.__n_samples


    def scramble_data(self, scramble_mode, seed = 0):
        '''
        Mix up the samples using the specified mode:
            "atac"      - recombine only atac track w.r.t. the target gene
            "dna"       - recombine only dna sequence w.r.t. the target gene
            "pairwise"  - recombine both inputs, together, w.r.t. the target gene
            "separate"  - recombine each input, independently, w.r.t. the target gene
        '''
        np.random.seed(seed)
        n = self.__n_samples
        idx = np.arange(n)
        np.random.shuffle(idx)
        if scramble_mode == 'atac':
            a = self.__torch_atac
            self.__torch_atac = a[idx]
        elif scramble_mode == 'dna':
            p = self.__torch_promoters
            self.__torch_promoters = p[idx]
        elif scramble_mode == 'pairwise':
            p = self.__torch_promoters
            a = self.__torch_atac
            self.__torch_promoters = p[idx]
            self.__torch_atac = a[idx]
        elif scramble_mode == 'separate':
            p = self.__torch_promoters
            a = self.__torch_atac
            self.__torch_promoters = p[idx]
            np.random.seed(seed + 1000)
            idx = np.arange(n)
            np.random.shuffle(idx)
            self.__torch_atac = a[idx]
        else:
            print('Error: invalid scramble mode given ({})'.format(scramble_mode))
            sys.exit(2)

    

    def make_train_val_split(self, split_type, seed = 0):
        if split_type == 'pooled':
            self.__train_indices, self.__val_indices = seq2exp_functions.ignorant_train_val_split(self.__data['samples'], val_prop = 0.25, seed = seed)
        else:
            split_fn = getattr(seq2exp_functions, split_type + '_train_val_split')
            self.__train_indices, self.__val_indices = split_fn(self.__data['samples'], val_prop = 0.25, seed = seed)


    def make_dataloader(self, batch_size = 1024, shuffle = True, num_workers = 1, seed = 0, hv_idx = None):
        torch.manual_seed(seed)
        target = self.__torch_rna
        n = len(target.shape)
        if self.__train_indices is None:
            if hv_idx is None:
                data = torch.utils.data.TensorDataset(self.__torch_promoters.float(), self.__torch_atac.float(), target.float())
                dl = torch.utils.data.DataLoader(data, batch_size = batch_size, num_workers = num_workers, shuffle = shuffle)
                del data
                return dl
            else:
                data = torch.utils.data.TensorDataset(self.__torch_promoters.float()[hv_idx], self.__torch_atac.float()[hv_idx], target.float()[hv_idx])
                dl = torch.utils.data.DataLoader(data, batch_size = batch_size, num_workers = num_workers, shuffle = shuffle)
                del data
                return dl
        else:
            train_data = torch.utils.data.TensorDataset(self.__torch_promoters[self.__train_indices].float(), self.__torch_atac[self.__train_indices].float(), target[self.__train_indices].float())
            train_dl = torch.utils.data.DataLoader(train_data, batch_size = batch_size, num_workers = num_workers, shuffle = shuffle)
            del train_data
            val_data = torch.utils.data.TensorDataset(self.__torch_promoters[self.__val_indices].float(), self.__torch_atac[self.__val_indices].float(), target[self.__val_indices].float())
            val_dl = torch.utils.data.DataLoader(val_data, batch_size = batch_size, num_workers = num_workers, shuffle = shuffle)
            del val_data
            return train_dl, val_dl
