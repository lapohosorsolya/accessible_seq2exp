import os, sys, gc, getopt, inspect, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
from tqdm import tqdm
from deeplift.dinuc_shuffle import dinuc_shuffle
from seq2exp_functions import *
from models import AugmentedSeq2ExpSequential


'''
USAGE NOTES

Required Inputs
---------------
    -t (path to directory with trained models)
        the full path to the directory containing models trained on DNA, ATAC, or both
    -d (path to directory with test data)
        the full path to the directory containing test data
    -o (output directory)
        the directory where the output should be written

'''


def main(argv):
    try:
        opts, _ = getopt.getopt(argv, 't:d:o:')
    except getopt.GetoptError:
        print('Error: cannot parse command line inputs')
        sys.exit(2) 
    for opt, arg in opts:
        if opt == '-t':
            global trained_model_dir
            trained_model_dir = arg
        elif opt == '-d':
            global data_dir
            data_dir = arg
        elif opt == '-o':
            global output_dir
            output_dir = arg


if __name__ == "__main__":

    main(sys.argv[1:])
    drop = 0.5
    in_len = 2000
    batch_size = 512
    seeds = [7, 25, 39, 101, 144]
    device = torch.device('cuda:6')

    # check if input files and directories exist
    check_path(trained_model_dir, dir = True)
    check_path(data_dir, dir = True)
    check_path(output_dir, dir = True)

    train_cell_code = os.path.basename(trained_model_dir)

    # make new output directories
    output_subdir = os.path.join(output_dir, '{}_shapley_values'.format(train_cell_code))
    full_output_dir = os.path.join(output_subdir, 'full_models')
    dna_output_dir = os.path.join(output_subdir, 'dna_models')
    if os.path.isdir(output_subdir):
        print('Output directory already exists:')
        print(output_subdir)
        sys.exit(2)
    else:
        os.mkdir(output_subdir)
        os.mkdir(full_output_dir)
        os.mkdir(dna_output_dir)

    # get a list of model subdirectories and their hyperparameters (only those with DNA channels)
    dir_prefix_set = set(sorted([ '_'.join(d.split('_')[:-1]) for d in os.listdir(trained_model_dir) ]))
    model_dict = {}
    for d in dir_prefix_set:
        with open(trained_model_dir + '/' + d + '_logs/hyperparameters.json', 'r') as f:
            params = json.load(f)
        include_dna = params['include_dna']
        include_atac = params['include_atac']
        if include_dna == True:
            if include_atac == True:
                model_key = 'dna_atac'
            else:
                model_key = 'dna'
            model_dict[model_key] = { 'model_path': trained_model_dir + '/' + d + '_models', 'include_dna': include_dna, 'include_atac': include_atac }

    # loop through CV folds
    for fold in range(5):

        print('Fold #{}'.format(fold + 1))

        # load test data; use only the first 100 genes to speed up
        test_data = np.load(data_dir + '/test_{}.npz'.format(fold))
        test_gene_ids, test_promoter_data, test_atac_data, test_rna_data = fetch_torch_samples(test_data)
        test_samples_full = torch.concatenate([test_promoter_data.float(), test_atac_data.reshape((len(test_gene_ids), 1, 2000)).float()], dim = 1)
        test_samples_dna = test_promoter_data.float()

        # loop through random seeds
        for i in range(5):

            print('Seed #{}'.format(i + 1))

            # get 100 background samples from train data
            train_data = np.load(data_dir + '/train_{}.npz'.format(fold))
            gene_ids, promoter_data, atac_data, rna_data = fetch_torch_samples(train_data)

            # full model
            if 'dna_atac' in model_dict.keys():

                print('Explaining full model...')

                # load trained model state and convert to nn.Sequential format for SHAP
                state_full = torch.load(model_dict['dna_atac']['model_path'] + '/{}.{}.pt'.format(fold, i), map_location = 'cpu')
                orig_names_full = []
                for l, layer in enumerate(state_full):
                    orig_names_full.append(layer)
                model_full_seq = AugmentedSeq2ExpSequential(dropout = drop, input_length = in_len, atac_channel = True, dna_channels = True)
                new_state = model_full_seq.state_dict()
                copied_state = new_state.copy()
                for l, layer in enumerate(new_state):
                    copied_state[layer] = state_full[orig_names_full[l]]
                model_full_seq.load_state_dict(copied_state)
                model_full_seq.eval()

                # explain each sequence
                shap_values_full = np.zeros(test_samples_full.shape)
                for s, seq in tqdm(enumerate(test_samples_full), total = test_samples_full.shape[0]):

                    # generate dinucleotide shuffled background
                    background_arrs = dinuc_shuffle(flip_gc(np.array(test_samples_dna[s])).T, num_shufs = 100)
                    background_dna = torch.Tensor(np.array([ flip_gc(arr.T) for arr in background_arrs ])).float()
                    background_atac = torch.Tensor(np.array([ np.random.choice(atac_data.flatten(), 2000) for i in range(100) ])).float()
                    background_dna_atac = torch.cat((background_dna, background_atac.reshape((100, 1, 2000))), dim = 1)

                    # explain sample
                    full_sample_to_explain = seq.reshape((1, 5, 2000))
                    full_explainer = shap.DeepExplainer(model_full_seq, background_dna_atac)
                    full_explanation = full_explainer.shap_values(full_sample_to_explain, check_additivity = False)
                    shap_values_full[s] = np.squeeze(full_explanation)

                # save SHAP values
                np.save(os.path.join(full_output_dir, '{}.{}.npy'.format(fold, i)), shap_values_full)
                
            # DNA only model
            if 'dna' in model_dict.keys():

                print('Explaining DNA only model...')

                # load trained model state and convert to nn.Sequential format for SHAP
                state_dna = torch.load(model_dict['dna']['model_path'] + '/{}.{}.pt'.format(fold, i), map_location = 'cpu')
                orig_names_dna = []
                for l, layer in enumerate(state_dna):
                    orig_names_dna.append(layer)
                model_dna_seq = AugmentedSeq2ExpSequential(dropout = drop, input_length = in_len, atac_channel = False, dna_channels = True)
                new_state = model_dna_seq.state_dict()
                copied_state = new_state.copy()
                for l, layer in enumerate(new_state):
                    copied_state[layer] = state_dna[orig_names_dna[l]]
                model_dna_seq.load_state_dict(copied_state)
                model_dna_seq.eval()

                # explain each sequence
                shap_values_dna = np.zeros(test_samples_dna.shape)
                for s, seq in tqdm(enumerate(test_samples_dna), total = test_samples_dna.shape[0]):

                    # generate dinucleotide shuffled background
                    background_arrs = dinuc_shuffle(flip_gc(np.array(seq)).T, num_shufs = 100)
                    background_dna = torch.Tensor(np.array([ flip_gc(arr.T) for arr in background_arrs ])).float()

                    # explain sample
                    dna_sample_to_explain = seq.reshape((1, 4, 2000))
                    dna_explainer = shap.DeepExplainer(model_dna_seq, background_dna)
                    dna_explanation = dna_explainer.shap_values(dna_sample_to_explain, check_additivity = False)
                    shap_values_dna[s] = np.squeeze(dna_explanation)

                # save SHAP values
                np.save(os.path.join(dna_output_dir, '{}.{}.npy'.format(fold, i)), shap_values_dna)

    print('Finished.')