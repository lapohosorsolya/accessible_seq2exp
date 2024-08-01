import os, sys, gc, getopt, inspect
import numpy as np
import torch
import torch.nn as nn
from seq2exp_functions import *
from multiome import MultiOmicDataset
from seq2exp_training import regressor_prediction
import models
import pandas as pd
import json

'''
USAGE NOTES

Required Inputs
---------------
    -m (model name)
        the name of the model to train
    -t (path to directory with trained models)
        the full path to the directory containing models trained on DNA, ATAC, or both
    -d (path to directory with test data)
        the full path to the directory containing test data
    -o (output directory)
        the directory where the output should be written

Optional Inputs
---------------
    -v (path to text file with highly variable genes)
        if provided, evaluation is done only with highly variable genes

'''


def main(argv):
    try:
        opts, _ = getopt.getopt(argv, 'm:t:d:o:v:')
    except getopt.GetoptError:
        print('Error: cannot parse command line inputs')
        sys.exit(2) 
    for opt, arg in opts:
        if opt == '-m':
            global model_name
            model_name = arg
        elif opt == '-t':
            global trained_model_dir
            trained_model_dir = arg
        elif opt == '-d':
            global test_data_dir
            test_data_dir = arg
        elif opt == '-o':
            global output_dir
            output_dir = arg
        elif opt == '-v':
            global hv_path
            hv_path = arg


if __name__ == "__main__":

    
    drop = 0.5
    in_len = 2000
    batch_size = 512
    hv_path = None
    hv_genes = None
    main(sys.argv[1:])

    # check if model name is valid
    if model_name not in [ i[0] for i in inspect.getmembers(models, inspect.isclass) ]:
        print('I do not know this model: {}'.format(model_name))
        sys.exit(2)
    model_class = getattr(models, model_name)

    # check if input files and directories exist
    check_path(trained_model_dir, dir = True)
    check_path(test_data_dir, dir = True)
    check_path(output_dir, dir = True)
    if hv_path is not None:
        check_path(hv_path, dir = False)
        hv_genes = np.loadtxt(hv_path, dtype = str)

    # get names of train and test cell types
    train_cell_code = os.path.basename(trained_model_dir)
    test_cell_code = os.path.basename(test_data_dir)
    if hv_path is not None:
        test_cell_code += '-hv'

    # get a list of model subdirectories and their hyperparameters
    dir_prefix_set = set(sorted([ '_'.join(d.split('_')[:-1]) for d in os.listdir(trained_model_dir) ]))
    model_dict = {}
    for d in dir_prefix_set:
        with open(trained_model_dir + '/' + d + '_logs/hyperparameters.json', 'r') as f:
            params = json.load(f)
        include_dna = params['include_dna']
        include_atac = params['include_atac']
        if 'shuffle_mode' in params.keys():
            if params['shuffle_mode'] is not None:
                continue
        if include_dna == True:
            if include_atac == True:
                model_key = 'dna_atac'
            else:
                model_key = 'dna'
        else:
            model_key = 'atac'
        model_dict[model_key] = { 'model_path': trained_model_dir + '/' + d + '_models', 'include_dna': include_dna, 'include_atac': include_atac }
    
    # evaluate
    result_dict = {}
    for key in model_dict.keys():
        print(model_dict[key]['model_path'])
        fold_pred_means = []
        fold_pred_stds = []
        fold_true = []
        fold_metrics = {}
        for fold in range(5):
            # get data
            modata = MultiOmicDataset(test_data_dir + '/test_{}.npz'.format(fold))
            samples = modata.fetch_torch_samples(return_samples = True)
            gene_ids = samples[0]
            n_samples = modata.get_n_samples()
            # get indices of highly variable genes if needed
            if hv_path is not None:
                idxs = [ i for i in range(n_samples) if gene_ids[i] in hv_genes ]
                test_dl = modata.make_dataloader(batch_size = batch_size, shuffle = False, num_workers = 1, seed = 0, hv_idx = idxs)
                n_samples = len(idxs)
            else:
                test_dl = modata.make_dataloader(batch_size = batch_size, shuffle = False, num_workers = 1, seed = 0)
            # collect predictions
            true_exp_seeds = np.zeros((5, n_samples))
            pred_exp_seeds = np.zeros((5, n_samples))
            for i in range(5):
                # load model
                model = model_class(dropout = drop, input_length = in_len, atac_channels = model_dict[key]['include_atac'], dna_channels = model_dict[key]['include_dna'])
                model.load_state_dict(torch.load(model_dict[key]['model_path'] + '/{}.{}.pt'.format(fold, i), map_location = 'cpu'))
                # test the trained model
                true_exp, pred_exp = regressor_prediction(model, test_dl, device = torch.device('cpu'), batch_size = batch_size, num_workers = 1)
                true_exp_seeds[i] = true_exp.flatten()
                pred_exp_seeds[i] = pred_exp.flatten()
            # calculate mean, std, and metrics
            preds_mean = np.mean(pred_exp_seeds, axis = 0)
            preds_std = np.std(pred_exp_seeds, axis = 0)
            metrics = calculate_regressor_metrics(preds_mean, true_exp_seeds[0])
            # add to fold collection
            fold_pred_means.append(preds_mean)
            fold_pred_stds.append(preds_std)
            fold_true.append(true_exp_seeds[0])
            fold_metrics[fold] = metrics
        result_dict[key] = { 'fold_metrics': fold_metrics, 'fold_pred_means': fold_pred_means, 'fold_pred_stds': fold_pred_stds, 'fold_true': fold_true }

    # save results
    for key in result_dict.keys():
        # dataframe of results
        performance_df = pd.DataFrame.from_dict(result_dict[key]['fold_metrics'], orient = 'index')
        performance_df.to_csv(output_dir + '/{}_performance_on_{}_{}.csv'.format(train_cell_code, test_cell_code, key))

    # summarize results
    df_list = []
    for key in ['dna', 'atac', 'dna_atac']:
        df_list.append(pd.read_csv(output_dir + '/{}_performance_on_{}_{}.csv'.format(train_cell_code, test_cell_code, key), index_col = 0))
    summary_df = pd.DataFrame(columns = df_list[0].columns)
    summary_df.loc['DNA mean'] = df_list[0].mean(axis = 0)
    summary_df.loc['DNA std'] = df_list[0].std(axis = 0)
    summary_df.loc['ATAC mean'] = df_list[1].mean(axis = 0)
    summary_df.loc['ATAC std'] = df_list[1].std(axis = 0)
    summary_df.loc['DNA + ATAC mean'] = df_list[2].mean(axis = 0)
    summary_df.loc['DNA + ATAC std'] = df_list[2].std(axis = 0)
    summary_df.to_csv(output_dir + '/{}_performance_on_{}_summary.csv'.format(train_cell_code, test_cell_code))
