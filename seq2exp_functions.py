import os, sys, math
import kneed
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split



def set_plot_style(font_path, style_path):
    mpl.font_manager.fontManager.addfont(font_path)
    plt.style.use(style_path)


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


def get_promoter(first_exon):
    '''
    Get the location of promoter given the first exon.
    '''
    chrom, start, end, strand = first_exon
    exon_start = int(start)
    exon_end = int(end)
    if strand == '+':
        promoter_end = exon_start - 1
        promoter_start = promoter_end - 500
        promoter = [chrom, promoter_start, promoter_end, strand]
    elif strand == '-':
        promoter_start = exon_end + 1
        promoter_end = promoter_start + 500
        promoter = [chrom, promoter_start, promoter_end, strand]
    else:
        promoter = None
    return promoter


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


def check_if_upstream(exon1, exon2):
    '''
    Check if exon 2 is upstream of exon 1.
    '''
    _, start1, end1, strand1 = exon1
    _, start2, end2, strand2 = exon2
    if strand1 != strand2:
        upstream = None
    else:
        if strand1 == '+':
            upstream = start2 < start1
        elif strand1 == '-':
            upstream = end2 > end1
        else:
            upstream = None
    return upstream


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

# ======================

def idx_to_seq(idx_arr):
    base_idx = { 0: 'A', 1: 'G', 2: 'C', 3: 'T' }
    seq = ''
    for i in idx_arr:
        seq += base_idx[i]
    return seq


def get_pwm(arr, background, add_pseudocount = True):
    '''
    Get the position weight matrix (PWM) from a position frequency matrix (PFM).

    Parameters
    ----------
    arr : numpy array
        position-frequency matrix (numpy-formatted JASPAR motif with rows corresponding to {A, G, C, T})
    background : numpy array
        background frequencies of each nucleotide {A, G, C, T}
    add_pseudocount : bool
        whether to add a count of 1 to every element of the PFM (avoids taking the log of 0)

    Returns
    -------
    pwm : numpy array
        the position-weight matrix
    '''
    if add_pseudocount:
        arr = np.add(arr, 1)
    prob = np.divide(arr, arr.sum(axis = 0))
    weights = np.divide(prob, np.repeat(background, arr.shape[1], axis = 1))
    pwm = -np.log2(weights)
    return pwm


def get_min_max_motif_scores(pwm):
    '''
    Get the minimum and maximum possible scores of a position weight matrix (PWM).

    Parameters
    ----------
    pwm : numpy array
        position-weight matrix

    Returns
    -------
    min_score : float
        minimum possible score
    max_score : float
        maximum possible score
    '''
    max_n = pwm.max(axis = 0)
    min_n = pwm.min(axis = 0)
    max_score = max_n.sum()
    min_score = min_n.sum()
    return min_score, max_score


def get_pwm_info(pwm_dir, pwm_files):
    '''
    Get the lengths, min scores, and max scores for a list of PWMs.

    Parameters
    ----------
    pwm_dir : str
        path of the directory containing numpy-formatted PWMs
    pwm_files : list of str
        filenames of the PWMs

    Returns
    -------
    lengths : list of int
        lengths of the PWMs
    min_scores : list of float
        minimum possible scores
    max_scores : list of float
        maximum possible scores
    '''
    lengths = []
    min_scores = []
    max_scores = []
    for file in pwm_files:
        pwm = np.load(pwm_dir + '/' + file)
        lengths.append(pwm.shape[1])
        mi, ma = get_min_max_motif_scores(pwm)
        min_scores.append(mi)
        max_scores.append(ma)
    return lengths, min_scores, max_scores


def score_sequence(seq, pwm):
    '''
    Score a sequence and its reverse complement against a PWM. The leftmost bases in the sequence are used for scoring. Only the higher score is returned.

    Parameters
    ----------
    seq : str
        the sequence of nucleotides to be scored (must be at least as long as the PWM)
    pwm : numpy array
        the position-weight matrix (PWM) to be used

    Returns
    -------
    max_score : int
        the best log-likelihood score of the sequence and its reverse complement
    '''
    d = { 'A': 0, 'G': 1, 'C': 2, 'T': 3 }
    l = pwm.shape[1]
    rev_comp = get_reverse_complement(seq[:l])
    # convert letters to numbers
    idx1 = [ d[i] for i in seq ]
    idx2 = [ d[i] for i in rev_comp ]
    # check nucleotides at the start of the sequence
    score1 = []
    score2 = []
    for i in range(l):
        score1.append(pwm[idx1[i], i])
        score2.append(pwm[idx2[i], i])
    max_score = max([sum(score1), sum(score2)])
    return max_score


# def score_sequence_strand(seq, pwm):
#     '''
#     Score a sequence and its reverse complement against a PWM. The leftmost bases in the sequence are used for scoring. Only the higher score is returned.

#     Parameters
#     ----------
#     seq : str
#         the sequence of nucleotides to be scored (must be at least as long as the PWM)
#     pwm : numpy array
#         the position-weight matrix (PWM) to be used

#     Returns
#     -------
#     max_score : int
#         the best log-likelihood score of the sequence and its reverse complement
#     '''
#     d = { 'A': 0, 'G': 1, 'C': 2, 'T': 3 }
#     l = pwm.shape[1]
#     rev_comp = get_reverse_complement(seq[:l])
#     # convert letters to numbers
#     idx1 = [ d[i] for i in seq ]
#     idx2 = [ d[i] for i in rev_comp ]
#     # check nucleotides at the start of the sequence
#     score1 = []
#     score2 = []
#     for i in range(l):
#         score1.append(pwm[idx1[i], i])
#         score2.append(pwm[idx2[i], i])
#     max_score = max([sum(score1), sum(score2)])
#     return max_score



def normalize(X, x_min, x_max):
    """
    Scale motif scores using their min and max scores.

    Parameters
    ----------
    X : numpy array
        array to be normalized
    x_min : float
        lower bound
    x_max : float
        upper bound

    Returns
    -------
    X : numpy array
        normalized input array
    """
    denominator = x_max - x_min
    numerator = X - x_min
    return numerator / denominator


# jaspar motif format: A, C, G, T base order ==> change to A, G, C, T order for numpy matrices
def extract_jaspar_pfm(lines):
    '''
    Convert jaspar 
    '''
    mat = []
    l = []
    for i in range(1, 5):
        tokens = lines[i].split()
        nums = []
        for t in tokens:
            if t.isnumeric():
                nums.append(int(t))
        mat.append(nums)
        l.append(len(nums))
    if sum(l)/4 != l[0]:
        print('Row lengths do not match!')
        print(l)
        length = 0
        arr = None
    else:
        length = l[0]
        arr = np.array([mat[0], mat[2], mat[1], mat[3]])
    return length, arr


# ======================

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


def ignorant_train_val_split(samples, val_prop = 0.25, seed = 0):
    '''
    Perform an ignorant split on cell and gene indices.
    '''
    rnd = np.random.RandomState(seed)
    train_indices, val_indices = train_test_split(np.arange(samples.shape[0]), test_size = val_prop, shuffle = True, random_state = rnd)
    return train_indices, val_indices


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


def calculate_classifier_metrics(y_hat, y):
    fpr, tpr, _ = roc_curve(y, y_hat, pos_label = 1)
    auroc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y, y_hat, pos_label = 1)
    aupr = average_precision_score(y, y_hat, pos_label = 1)
    base_aupr = y.sum().item() / y.shape[0]
    return { 'fpr': fpr, 'tpr': tpr, 'auroc': auroc, 'precision': precision, 'recall': recall, 'aupr': aupr, 'base_aupr': base_aupr }


def calculate_regressor_metrics(y_hat, y):
    print(min(y_hat), max(y_hat))
    print(min(y), max(y))
    pearson_result = pearsonr(y, y_hat)
    spearman_result = spearmanr(y, y_hat)
    mse = mean_squared_error(y, y_hat)
    r2 = r2_score(y, y_hat)
    return { 'pearson_r': pearson_result.statistic, 'pearson_p': pearson_result.pvalue, 'spearman_r': spearman_result.statistic, 'spearman_p': spearman_result.pvalue, 'mse': mse, 'r2': r2 }


def find_2_knees(data):
    # round 1 of knee point selection
    sorted_data = np.flip(np.sort(data))
    kl = kneed.KneeLocator(np.arange(sorted_data.shape[0]), sorted_data, curve = 'convex', direction = 'decreasing')
    # find max difference
    max_y = max(kl.y_difference)
    max_i = np.where(kl.y_difference == max_y)[0][0]
    max_x = kl.x_difference[max_i]
    # scale back to original
    x_cutoff_1 = round((data.shape[0]-1) * max_x)
    y_cutoff_1 = sorted_data[x_cutoff_1]
    x_1 = sorted_data.shape[0] - x_cutoff_1

    # round 2 of knee point selection
    updated_sorted_data = sorted_data[sorted_data < y_cutoff_1]
    kl = kneed.KneeLocator(np.arange(updated_sorted_data.shape[0]), updated_sorted_data, curve = 'convex', direction = 'decreasing')
    # find max difference
    max_y = max(kl.y_difference)
    max_i = np.where(kl.y_difference == max_y)[0][0]
    max_x = kl.x_difference[max_i]
    # scale back to original
    x_cutoff_0 = round((updated_sorted_data.shape[0]-1) * max_x)
    y_cutoff_0 = updated_sorted_data[x_cutoff_0]
    x_0 = updated_sorted_data.shape[0] - x_cutoff_0
    return y_cutoff_0, y_cutoff_1


def plot_loss(train_epochs, val_epochs, train_loss, val_loss, ax):
    ax.plot(train_epochs, train_loss, color = 'b', label = 'training')
    ax.plot(val_epochs, val_loss, color = 'r', label = 'validation')
    ax.legend(frameon = False, fontsize = 8)
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.set_title('loss over epochs')


def plot_training_loss_components(train_epochs, train_loss, ax, components):
    colors = ['#03045e', '#00b4d8', '#0077b6']
    linestyles = [':', '-.', '--']
    for i in range(len(components)):
        ax.plot(train_epochs, train_loss[i], color = colors[i], linestyle = linestyles[i], label = components[i])
    ax.legend(frameon = False, fontsize = 8)
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.set_title('training loss components')


def plot_roc_curve_single(fpr, tpr, auroc, ax):
    ax.plot((0, 1), (0, 1), color = 'k', linestyle = '--', linewidth = 1)
    ax.plot(fpr, tpr, color = 'k', label = 'AUROC = {0:.3f}'.format(auroc))
    ax.set_xlim(xmin = 0, xmax = 1)
    ax.set_ylim(ymin = 0, ymax = 1)
    ax.legend(frameon = False, loc = 'lower right', fontsize = 8)
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    ax.set_title('ROC curve')


def plot_roc_curve_multiple(fprs, tprs, aurocs, ax, colors):
    ax.plot((0, 1), (0, 1), color = 'k', linestyle = '--', linewidth = 1)
    for i in range(len(tprs)):
        ax.plot(fprs[i], tprs[i], color = colors[i], alpha = 0.5, label = 'AUROC = {0:.3f}'.format(aurocs[i]))
    ax.set_xlim(xmin = 0, xmax = 1)
    ax.set_ylim(ymin = 0, ymax = 1)
    ax.legend(frameon = False, loc = 'lower right', fontsize = 8)
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    ax.set_title('ROC curve')


def plot_pr_curve_single(precision, recall, aupr, base_aupr, ax):
    ax.plot((0, 1), (base_aupr, base_aupr), color = 'k', linestyle = '--', linewidth = 1)
    ax.plot(recall, precision, color = 'k', label = 'AUPR = {0:.3f}'.format(aupr))
    ax.set_xlim(xmin = 0, xmax = 1)
    ax.set_ylim(ymin = 0, ymax = 1)
    ax.legend(frameon = False, loc = 'upper right', fontsize = 8)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('PR curve')


def plot_pr_curve_multiple(precisions, recalls, auprs, base_aupr, ax, colors):
    ax.plot((0, 1), (base_aupr, base_aupr), color = 'k', linestyle = '--', linewidth = 1)
    for i in range(len(precisions)):
        ax.plot(recalls[i], precisions[i], color = colors[i], alpha = 0.5, label = 'AUPR = {0:.3f}'.format(auprs[i]))
    ax.set_xlim(xmin = 0, xmax = 1)
    ax.set_ylim(ymin = 0, ymax = 1)
    ax.legend(frameon = False, loc = 'upper right', fontsize = 8)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('PR curve')


def plot_roc_curve_multiclass(fprs, tprs, aurocs, labels, colors, ax):
    ax.plot((0, 1), (0, 1), color = 'k', linestyle = '--', linewidth = 1)
    for i in range(len(labels)):
        ax.plot(fprs[i], tprs[i], color = colors[i], label = labels[i] + '\nAUROC = {0:.3f}'.format(aurocs[i]))
    ax.set_xlim(xmin = 0, xmax = 1)
    ax.set_ylim(ymin = 0, ymax = 1)
    ax.legend(frameon = False, fontsize = 8, loc = 'lower right')
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    ax.set_title('ROC curve (one-vs-rest)')


def plot_roc_curve_cv(roc_splits_exp, aurocs, ax):
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    for split in range(len(roc_splits_exp)):
        fpr, tpr = roc_splits_exp[split]
        ax.plot(fpr, tpr, color = 'k', alpha = 0.2)
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis = 0)
    ax.plot(base_fpr, mean_tprs, color = 'k', alpha = 1, label = 'mean AUROC = {0:.3f}'.format(np.mean(aurocs)))
    ax.plot((0, 1), (0, 1), color = 'k', linestyle = '--', linewidth = 1)
    ax.set_xlim(xmin = 0, xmax = 1)
    ax.set_ylim(ymin = 0, ymax = 1)
    ax.legend(frameon = False, loc = 'lower right', fontsize = 8)
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    ax.set_title('ROC curve')


def plot_loss_cv(train_losses, val_losses, ax):
    for split in range(len(train_losses)):
        ax.plot(train_losses[split][0], train_losses[split][1], color = 'b', label = 'training', alpha = 0.4)
        ax.plot(val_losses[split][0], val_losses[split][1], color = 'r', label = 'validation', alpha = 0.4)
        if split == 0:
            ax.legend(frameon = False, fontsize = 8)
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.set_title('loss over epochs')


def plot_pred_vs_true_contour_with_vars(pred, true, filename):
    fig, axes = plt.subplots(1, 5, figsize = (16, 3), sharey = True, gridspec_kw = { 'wspace': 0.1 })
    labels = [0, 0.01, 0.05, 0.1, 0.2]
    intervals = np.linspace(0.1, 5, 9)
    for v, ax in enumerate(axes):
        x = true[v].flatten()
        y = pred[v].flatten()
        z = np.zeros((100, 100))
        for i in range(x.shape[0]):
            xi = int(np.round(x[i] + 0.005, 2) * 100)
            yi = int(np.round(y[i] + 0.005, 2) * 100)
            z[xi, yi] += 1
        blur = gaussian_filter(np.log1p(z), 3)
        xi = np.linspace(-0.01, 1, 101)
        yi = np.linspace(-0.01, 1, 101)
        closed_lines = np.zeros((101, 101))
        closed_lines[1:, 1:] = blur.T
        ax.contour(xi, yi, closed_lines, levels = intervals, linewidths = 1, cmap = 'OrRd')
        ax.contourf(xi, yi, closed_lines, levels = intervals, cmap = 'OrRd', alpha = 0.5)
        ax.set_xlim(xmin = -0.05, xmax = 1.05)
        ax.set_ylim(ymin = -0.05, ymax = 1.05)
        if v == 0:
            ax.set_ylabel('predicted')
        ax.set_xlabel('ground truth')
        ax.set_title(r'gene variance $\geq$ {}'.format(labels[v]))
    fig.savefig(filename, bbox_inches = 'tight')


def plot_metrics_with_vars(aurocs, auprs, base_auprs, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (6, 3))
    pos = range(5)
    labels = [0, 0.01, 0.05, 0.1, 0.2]
    mean_aurocs = np.mean(aurocs, axis = 1) # mean across 5 seeds
    mean_auprs = np.mean(auprs, axis = 1) # mean across 5 seeds
    mean_base_auprs = np.mean(base_auprs, axis = 1) # mean across 5 seeds
    ax1.bar(pos, np.mean(mean_aurocs, axis = 0), color = 'k', alpha = 0.7)
    ax1.errorbar(x = pos, y = np.mean(mean_aurocs, axis = 0), yerr = np.std(mean_aurocs, axis = 0), capsize = 5, fmt = 'none', ecolor = 'k', elinewidth = 1)
    ax1.set_ylim(ymin = 0.5, ymax = 1)
    ax1.set_ylabel('AUROC')
    ax1.set_xticks(pos)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel('variance cutoff')
    ax2.bar(pos, np.mean(mean_auprs, axis = 0), color = 'k', alpha = 0.7)
    ax2.errorbar(x = pos, y = np.mean(mean_auprs, axis = 0), yerr = np.std(mean_auprs, axis = 0), capsize = 5, fmt = 'none', ecolor = 'k', elinewidth = 1)
    ax2.bar(pos, np.mean(mean_base_auprs, axis = 0), color = 'lightgray')
    ax2.set_ylabel('AUPRC')
    ax2.set_xticks(pos)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel('variance cutoff')
    fig.tight_layout()
    fig.savefig(filename, bbox_inches = 'tight')


def calculate_pooling_output_length(L_in, padding, kernel_size, stride):
    L_out = (L_in + 2 * padding - kernel_size) /  stride + 1
    return math.floor(L_out)


def calculate_conv_output_length(L_in, padding, kernel_size, stride, dilation):
    L_out = (L_in + 2 * padding - (dilation * kernel_size - 1)) /  stride + 1
    return math.floor(L_out)


def make_contour_plot(ax, x, y, intervals):
    max_x = np.max(x)
    scale_x = max_x * 1.1
    max_y = np.max(y)
    scale_y = max_y * 1.1
    z = np.zeros((100, 100))
    for i in range(x.shape[0]):
        xi = int(round(x[i] / scale_x + 0.01, 2) * 100)
        yi = int(round(y[i] / scale_y + 0.01, 2) * 100)
        z[xi, yi] += 1
    blur = gaussian_filter(np.log1p(z), 3)
    xi = np.linspace(-(scale_x * 0.01), scale_x, 101)
    yi = np.linspace(-(scale_y * 0.01), scale_y, 101)
    closed_lines = np.zeros((101, 101))
    closed_lines[1:, 1:] = blur.T
    ax.contour(xi, yi, closed_lines, levels = intervals, linewidths = 1, cmap = 'magma_r')
    ax.contourf(xi, yi, closed_lines, levels = intervals, cmap = 'magma_r', alpha = 0.5)
    ax.set_xlim(xmin = 0 - (scale_x * 0.05), xmax = scale_x)
    ax.set_ylim(ymin = 0 - (scale_y * 0.05), ymax = scale_y)