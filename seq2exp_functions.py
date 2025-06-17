import os, sys, math
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score



def check_path(path, dir = False):
    if dir == False:
        check = os.path.isfile(path)
    else:
        check = os.path.isdir(path)
    if check == True:
        # print('\n::: Found {}'.format(path))
        pass
    else:
        print('\n::: {} does not exist'.format(path))
        sys.exit(2)


def encode_promoter(promoter):
    return torch.nn.functional.one_hot(torch.tensor(promoter), 4).T


def fetch_torch_samples(data, bases = 4):
    promoters = data['sequence']
    n = len(promoters)
    p = torch.zeros((n, bases, 2000), dtype = int)
    for i in tqdm(range(n), total = n):
        seq = promoters[i]
        p[i] = encode_promoter(seq)
    a = torch.from_numpy(data['atac'])
    r = torch.from_numpy(data['rna'])
    gene_ids = list(data['samples'])
    return gene_ids, p, a, r


def flip_gc(one_hot_seq):
    if type(one_hot_seq) == np.ndarray:
        new_seq = np.zeros(one_hot_seq.shape)
    elif type(one_hot_seq) == torch.Tensor:
        new_seq = torch.zeros(one_hot_seq.shape)
    new_seq[0] = one_hot_seq[0]
    new_seq[1] = one_hot_seq[2]
    new_seq[2] = one_hot_seq[1]
    new_seq[3] = one_hot_seq[3]
    return new_seq


def get_reverse_complement(seq):
    '''
    Get the reverse complement of a DNA sequence.
    '''
    complement = { 'A': 'T', 'G': 'C', 'T': 'A', 'C': 'G', 'N': 'N' }
    rev_comp = ''
    l = len(seq)
    for n in range(l):
        rev_comp = rev_comp + complement[seq[l - n - 1]]
    return rev_comp


def make_agct_one_hot(seq):
    '''
    Generate a one-hot numpy array of a DNA sequence.

    Parameters
    ----------
    seq : str
        the DNA sequence

    Returns
    -------
    one_hot : numpy array
        the one-hot array with rows corresponding to A, G, C, T
    '''
    l = len(seq)
    one_hot = np.zeros((4, l), dtype = int)
    base_idx = { 'A': 0, 'G': 1, 'C': 2, 'T': 3 }
    for i in range(l):
        base = seq[i]
        one_hot[base_idx[base], i] = 1
    return one_hot


def idx_to_seq(idx_arr):
    base_idx = { 0: 'A', 1: 'G', 2: 'C', 3: 'T' }
    seq = ''
    for i in idx_arr:
        seq += base_idx[i]
    return seq


def make_agct_numeric_vector(seq):
    '''
    Generate a numpy array of a DNA sequence. Key: { 'A': 0, 'G': 1, 'C': 2, 'T': 3 }

    Parameters
    ----------
    seq : str
        the DNA sequence

    Returns
    -------
    vec : numpy array
        numeric sequence representing the nucleotide sequence
    '''
    l = len(seq)
    vec = np.zeros(l, dtype = int)
    base_idx = { 'A': 0, 'G': 1, 'C': 2, 'T': 3 }
    for i in range(l):
        base = seq[i]
        vec[i] = base_idx[base]
    return vec


def split_genes_kf(n_folds, genes):
    '''
    Split only genes k-fold.
    '''
    kf = KFold(n_splits = n_folds, shuffle = True, random_state = 1)
    gene_splits = list(kf.split(genes))
    train_indices = []
    test_indices = []
    for i in range(n_folds):
        train_idx, test_idx = gene_splits[i]
        train_indices.append(genes[train_idx])
        test_indices.append(genes[test_idx])
    return train_indices, test_indices


def calculate_regressor_metrics(y_hat, y):
    print(min(y_hat), max(y_hat))
    print(min(y), max(y))
    pearson_result = pearsonr(y, y_hat)
    spearman_result = spearmanr(y, y_hat)
    mse = mean_squared_error(y, y_hat)
    r2 = r2_score(y, y_hat)
    return { 'pearson_r': pearson_result.statistic, 'pearson_p': pearson_result.pvalue, 'spearman_r': spearman_result.statistic, 'spearman_p': spearman_result.pvalue, 'mse': mse, 'r2': r2 }


def plot_loss(train_epochs, val_epochs, train_loss, val_loss, ax):
    ax.plot(train_epochs, train_loss, color = 'b', label = 'training')
    ax.plot(val_epochs, val_loss, color = 'r', label = 'validation')
    ax.legend(frameon = False, fontsize = 8)
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.set_title('loss over epochs')


def calculate_pooling_output_length(L_in, padding, kernel_size, stride):
    L_out = (L_in + 2 * padding - kernel_size) /  stride + 1
    return math.floor(L_out)


def calculate_conv_output_length(L_in, padding, kernel_size, stride, dilation):
    L_out = (L_in + 2 * padding - (dilation * kernel_size - 1)) /  stride + 1
    return math.floor(L_out)
