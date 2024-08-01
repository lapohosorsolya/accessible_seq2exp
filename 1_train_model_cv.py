import os, sys, gc, getopt, inspect, json
import numpy as np
import torch
import torch.nn as nn
import seq2exp_functions
from multiome import *
from seq2exp_logger import CVLogger
from seq2exp_training import train_regressor_with_early_stop, test_regressor
import models

'''
USAGE NOTES

Required Inputs
---------------
    -m (model name)
        the name of the model to train
    -i (input directory)
        the full path to the directory containing train and test data for 5-fold CV
    -o (output directory)
        the directory where the output should be written (will make this directory if it does not exist yet)
    -g (gpu id)
        CUDA GPU ID (integer)
        
Optional Inputs
---------------
    -s (settings)
        path to JSON file with model settings and hyperparameters (overrides defaults); example provided in example/settings.json
    -r (output input_directory prefix)
        if resuming a partially complete run, specify the output input_directory prefix
    -p (path to pretrained model)
        if weights should be loaded from a pretrained DNA-only model, specify the path to the models directory

'''


def main(argv):
    try:
        opts, _ = getopt.getopt(argv, 'm:i:o:s:g:r:p:')
    except getopt.GetoptError:
        print('\n::: Error: cannot parse command line inputs')
        sys.exit(2) 
    for opt, arg in opts:
        if opt == '-m':
            global model_name
            model_name = arg
        elif opt == '-i':
            global input_dir
            input_dir = arg
        elif opt == '-o':
            global output_dir
            output_dir = arg
        elif opt == '-g':
            global gpu_id
            gpu_id = arg
        elif opt == '-s':
            global settings_file
            settings_file = arg
        elif opt == '-r':
            global resume
            resume = True
            global resume_prefix
            resume_prefix = arg
        elif opt == '-p':
            global use_pretrained
            use_pretrained = True
            global pretrained_model_dir
            pretrained_model_dir = arg


if __name__ == '__main__':

    resume = False
    start_fold = 0
    start_seed_idx = 0
    seeds = [7, 25, 39, 101, 144]
    use_pretrained = False
    pretrained_model_dir = None
    loss_func_dict = { 'mse': nn.MSELoss() }
    settings_file = None
    modifiable_settings = ['include_dna', 'include_atac', 'batch_size', 'max_epochs', 'lr', 'wd', 'shuffle_mode']

    # fixed hyperparameters
    drop = 0.5
    opt = 'adam'
    lr = 0.00005
    wd = 0.001
    max_epochs = 500
    val_interval = 4
    batch_size = 512
    include_dna = True
    include_atac = True
    shuffle_mode = None
    loss_func = 'mse'

    main(sys.argv[1:])
    n_seeds = len(seeds)
    device = torch.device('cuda:{}'.format(gpu_id))

    # check if model name is valid
    if model_name not in [ i[0] for i in inspect.getmembers(models, inspect.isclass) ]:
        print('Cannot run model: {}'.format(model_name))
        sys.exit(2)
    model_class = getattr(models, model_name)

    # check if input files and directories exist
    seq2exp_functions.check_path(input_dir, dir = True)
    seq2exp_functions.check_path(output_dir, dir = True)

    # use provided settings
    if settings_file is not None:
        seq2exp_functions.check_path(settings_file)
        with open(settings_file, 'r') as f:
            settings = json.load(f)
            for key in settings.keys():
                if key in modifiable_settings:
                    globals()[key] = settings[key]
    
    # get all training sets
    train_files = sorted([ i for i in os.listdir(input_dir) if 'train' in i ])
    n_folds = len(train_files)

    param_dict = { 'drop': drop, 'opt': opt, 'lr': lr, 'wd': wd, 'max_epochs': max_epochs, 'val_interval': val_interval, 'n_folds': n_folds, 'seeds': seeds, 'batch_size': batch_size, 'gpu_id': gpu_id, 'device': torch.cuda.get_device_name(device), 'include_dna': include_dna, 'include_atac': include_atac, 'shuffle_mode': shuffle_mode, 'loss_fn': loss_func, 'pretrained_model_dir': pretrained_model_dir }

    # initialize logger and set the start folds/trials
    if resume == True:
        logger = CVLogger(model_name, { 'dna': include_dna, 'atac': include_atac }, output_dir, input_dir, n_folds, resume = True, resume_prefix = resume_prefix, shuffle_mode = shuffle_mode)
        # get the next seed idx
        prev_s = logger.get_seed_no()
        prev_f = logger.get_fold_no()
        if prev_s == n_seeds - 1:
            start_seed_idx = 0
            # get the next fold
            if prev_f == n_folds - 1:
                print('Error: cannot resume if already finished')
                sys.exit(2)
            else:
                start_fold = prev_f + 1
        else:
            start_seed_idx = prev_s + 1
            start_fold = prev_f
    else:
        logger = CVLogger(model_name, { 'dna': include_dna, 'atac': include_atac }, output_dir, input_dir, n_folds, shuffle_mode = shuffle_mode)
        logger.record_input_dir(input_dir)
        logger.record_hyperparameters(param_dict)
    

    # cross-validation loop
    for fold in range(start_fold, n_folds):

        logger.set_fold(fold)

        # get the seed to start at (for resuming)
        if fold == start_fold:
            start_inner = start_seed_idx
        else:
            start_inner = 0

        for s in range(start_inner, n_seeds):

            torch.manual_seed(seeds[s])
            logger.set_seed(s)

            # load train data
            fullpath = os.path.join(input_dir, 'train_{}.npz'.format(fold))
            modata = MultiOmicDataset(fullpath)
            modata.fetch_torch_samples()
            if shuffle_mode in ['dna', 'atac', 'pairwise', 'separate']:
                modata.scramble_data(scramble_mode = shuffle_mode)
            modata.make_train_val_split('pooled', seed = seeds[s])
            train_dl, val_dl = modata.make_dataloader(batch_size = batch_size, shuffle = True, num_workers = 1, seed = seeds[s])
            in_len = modata.get_seq_len()

            loss_fn = loss_func_dict[loss_func]

            # initialize model
            model = model_class(dropout = drop, input_length = in_len, dna_channels = include_dna, atac_channels = include_atac).to(device)

            # load weights from pretrained model if needed
            if use_pretrained == True:
                pretrained_state = torch.load(os.path.join(pretrained_model_dir, '{}.{}.pt'.format(fold, s)))
                copied_state = model.state_dict().copy()
                for l, layer in enumerate(pretrained_state):
                    if layer == 'conv_1.weight':
                        copied_state['conv_1.weight'][:, :4, :] = pretrained_state['conv_1.weight']
                    else:
                        copied_state[layer] = pretrained_state[layer]
                model.load_state_dict(copied_state)

            # train and evaluate model with fixed hyperparameters
            train_regressor_with_early_stop(model, train_dl, val_dl, logger, opt, lr, wd, max_epochs, val_interval, device, loss_fn)
            
            # load test data
            fullpath = os.path.join(input_dir, 'test_{}.npz'.format(fold))
            modata = MultiOmicDataset(fullpath)
            modata.fetch_torch_samples()
            test_dl = modata.make_dataloader(batch_size = batch_size, shuffle = False, num_workers = 1, seed = seeds[s])
            del modata

            # test the trained model
            test_regressor(model, test_dl, logger, batch_size = batch_size)
            del test_dl
            gc.collect()

    # finish
    logger.signal_completion()