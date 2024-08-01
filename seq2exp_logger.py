import os, sys
import pickle, json
import numpy as np
import pandas as pd
from datetime import datetime


class CVLogger():
    '''
    Class to log progress and results of model training with cross-validation and multiple random seeds.
    '''
    def __init__(self, model_name, input_channels, output_dir, input_dir, folds, resume = False, resume_prefix = None, shuffle_mode = None):

        # private string attributes
        self.__model_name = model_name
        self.__input_channels = input_channels
        self.__output_dir = output_dir
        self.__train_data = input_dir
        self.__resuming = resume
        self.__folds = folds
        self.__shuffle_mode = shuffle_mode

        # check if a previous run is being resumed
        if resume == True:
            if resume_prefix is not None:
                # use previous directories
                self.__save_str = os.path.join(self.__output_dir, resume_prefix)
            else:
                print('Error: need a directory prefix to resume run')
                sys.exit(2)
        else:
            # initialize new directories with the current time
            self.__date_time = datetime.now()
            self.__save_str = os.path.join(self.__output_dir, self.__date_time.strftime('%y%m%d-%H%M%S') + '_' + self.__model_name + '_' + os.path.basename(self.__train_data))

        # directories
        self.__logging_dir = self.__save_str + '_logs'
        self.__loss_data_dir = self.__save_str + '_loss'
        self.__metrics_data_dir = self.__save_str + '_metrics'
        self.__models_dir = self.__save_str + '_models'

        # log files
        self.__log_file = os.path.join(self.__logging_dir, '.log')
        self.__val_results_file = os.path.join(self.__logging_dir, 'valresults.txt')
        self.__test_results_file = os.path.join(self.__logging_dir, 'testresults.txt')
        self.__params_file = os.path.join(self.__logging_dir, 'hyperparameters.json')

        self.__current_fold_no = 0
        self.__current_seed_no = 0
        
        self.__best_epoch = 0
        self.__best_val_loss = 100.0
        self.__best_val_pearson = 0
        self.__best_val_r2 = 0

        self.__val_loss_epochs = []
        self.__train_loss_epochs = []

        # make new directories and files if needed
        if resume == True:
            # read the previous output files to determine the current folds
            val_df = pd.read_csv(self.__val_results_file, sep = '\t')
            last_row = val_df.iloc[-1, :]
            self.__current_fold_no = int(last_row.fold)
            self.__current_seed_no = int(last_row.seed)
            self.__signal_resumption()
        else:
            self.__make_files()

        # log PID
        pid = os.getpid()
        with open(self.__log_file, 'a') as f:
            f.write('\nPID: {}\n\n'.format(pid))


    def __make_files(self):
        os.mkdir(self.__logging_dir)
        os.mkdir(self.__loss_data_dir)
        os.mkdir(self.__metrics_data_dir)
        os.mkdir(self.__models_dir)
        with open(self.__log_file, 'w') as f:
            f.write('\n'.join(['Log Start: ' + self.__date_time.strftime('%y-%m-%d %H:%M:%S'), 'Model Name: ' + self.__model_name, 'Training Data: ' + self.__train_data, 'DNA Channels: ' + str(self.__input_channels['dna']), 'ATAC Channel: ' + str(self.__input_channels['atac']), 'Shuffle Mode: ' + str(self.__shuffle_mode), '\n']))
        with open(self.__val_results_file, 'w') as f:
            f.write('\t'.join(['fold', 'seed', 'val_loss', 'pearson_r', 'r2', 'early_stop_epoch']))
            f.write('\n')
        with open(self.__test_results_file, 'w') as f:
            f.write('\t'.join(['fold', 'seed', 'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p', 'mse', 'r2']))
            f.write('\n')
        

    def record_hyperparameters(self, param_dict):
        '''
        Save the hyperparameters.
        '''
        with open(self.__params_file, 'w') as f:
            json.dump(param_dict, f, indent = 4, sort_keys = True)


    def record_input_dir(self, input_dir):
        '''
        Record the directory containing the indices of the training data.
        '''
        dt = datetime.now().strftime('%y-%m-%d %H:%M:%S')
        with open(self.__log_file, 'a') as f:
            f.write('\n{} -- TRAINING DATA:'.format(dt))
            f.write('\n\t{}'.format(input_dir))
            f.write('\n')


    def set_fold(self, fold):
        '''
        Update the current fold.
        '''
        if fold >= 0 and fold < self.__folds:
            self.__current_fold_no = fold
            self.__log_fold()
        else:
            pass

    
    def set_seed(self, seed):
        '''
        Update the current seed.
        '''
        self.__current_seed_no = seed
        self.__log_seed()


    def __log_fold(self):
        '''
        Log the current fold.
        '''
        dt = datetime.now().strftime('%y-%m-%d %H:%M:%S')
        with open(self.__log_file, 'a') as f:
            f.write('\n{} -- SET FOLD:'.format(dt))
            f.write('\n\t{}'.format(self.__current_fold_no))
            f.write('\n')

    def __log_seed(self):
        '''
        Log the current seed.
        '''
        dt = datetime.now().strftime('%y-%m-%d %H:%M:%S')
        with open(self.__log_file, 'a') as f:
            f.write('\n{} -- SET SEED:'.format(dt))
            f.write('\n\t{}'.format(self.__current_seed_no))
            f.write('\n')


    def get_fold_no(self):
        '''
        Return the current outer fold.
        '''
        return self.__current_fold_no


    def get_seed_no(self):
        '''
        Return the current trial number.
        '''
        return self.__current_seed_no
    

    def save_val_results(self):
        '''
        Log new validation results:
            - fold
            - seed
            - training loss
            - validation loss
            - pearson r
            - r2
            - early stop epoch
        '''
        new_row = [self.__current_fold_no, self.__current_seed_no, self.__best_val_loss, self.__best_val_pearson, self.__best_val_r2, self.__best_epoch] 
        with open(self.__val_results_file, 'a') as f:
            f.write('\t'.join([ str(i) for i in new_row ]))
            f.write('\n')
        dt = datetime.now().strftime('%y-%m-%d %H:%M:%S')
        with open(self.__log_file, 'a') as f:
            f.write('\n{} -- VALIDATION RESULTS:'.format(dt))
            f.write('\n\tfold = {}, seed = {}, val_loss = {}, pearson_r = {}, r2 = {}, early_stop_epoch = {}'.format(*new_row))
            f.write('\n')


    def reset_loss_curves(self):
        '''
        Reset the variables storing the training and validation loss curves.
        '''
        self.__train_loss_epochs = []
        self.__val_loss_epochs = []


    def record_training_start(self):
        '''
        Record time in log file.
        '''
        dt = datetime.now().strftime('%y-%m-%d %H:%M:%S')
        with open(self.__log_file, 'a') as f:
            f.write('\n{} -- TRAINING:'.format(dt))


    def append_to_train_loss(self, epoch, train_loss):
        '''
        Append a new point to the training loss curve.
        '''
        self.__train_loss_epochs.append([epoch, train_loss])


    def append_to_val_loss(self, epoch, val_loss, save_to_log = False):
        '''
        Append a new point to the validation loss curve.
        '''
        self.__val_loss_epochs.append([epoch, val_loss])
        if save_to_log == True:
            with open(self.__log_file, 'a') as f:
                f.write('\n\tepoch = {0}\t\ttrain_loss = {1:.4f}\t\tval_loss = {2:.4f}'.format(self.__val_loss_epochs[-1][0], self.__train_loss_epochs[-1][1], self.__val_loss_epochs[-1][1]))


    def save_loss_curve_data(self):
        '''
        Write the training and validation loss curves to file.
        '''
        loss_data = {}
        loss_data['train'] = np.array(self.__train_loss_epochs).T
        loss_data['val'] = np.array(self.__val_loss_epochs).T
        np.savez(os.path.join(self.__loss_data_dir, '{}.{}.npz'.format(self.__current_fold_no, self.__current_seed_no)), **loss_data)
        dt = datetime.now().strftime('%y-%m-%d %H:%M:%S')
        with open(self.__log_file, 'a') as f:
            f.write('\n{} -- SAVED LOSS CURVE DATA:'.format(dt))
            f.write('\n\tfold = {}, seed = {}'.format(self.__current_fold_no, self.__current_seed_no))
            f.write('\n')


    def record_final_train_metrics(self, pearson_r, r2):
        '''
        Write the training metrics from the last epoch to the log file.
        '''
        dt = datetime.now().strftime('%y-%m-%d %H:%M:%S')
        with open(self.__log_file, 'a') as f:
            f.write('\n')
            f.write('\n{} -- TRAINING COMPLETE!'.format(dt))
            f.write('\n\tlast epoch train pearson_r = {}, r2 = {}'.format(pearson_r, r2))
            f.write('\n')


    def save_test_metrics(self, metrics):
        '''
        Write test metrics to file. Should use output of calculate_regressor_metrics() from the seq2exp_functions module.
        '''
        with open(os.path.join(self.__metrics_data_dir, '{}.{}_metrics.pkl'.format(self.__current_fold_no, self.__current_seed_no)), 'wb') as f:
            pickle.dump(metrics, f)
        self.__log_test_results(metrics)


    def get_model_save_path(self):
        '''
        Return the full path with a unique filename corresponding to the current outer fold, parameter trial, and inner fold.
        '''
        return os.path.join(self.__models_dir, '{}.{}.pt'.format(self.__current_fold_no, self.__current_seed_no))


    def __log_test_results(self, metrics):
        '''
        Log model performance on the held-out test set:
            - pearson r
            - pearson p
            - spearman r
            - spearman p
            - mse
            - r2
        '''
        new_row = [self.__current_fold_no, self.__current_seed_no, metrics['pearson_r'], metrics['pearson_p'], metrics['spearman_r'], metrics['spearman_p'], metrics['mse'], metrics['r2']]
        with open(self.__test_results_file, 'a') as f:
            f.write('\t'.join([ str(i) for i in new_row ]))
            f.write('\n')
        dt = datetime.now().strftime('%y-%m-%d %H:%M:%S')
        with open(self.__log_file, 'a') as f:
            f.write('\n{} -- TEST RESULTS:'.format(dt))
            f.write('\n\tfold = {}, seed = {}, pearson_r = {}, pearson_p = {}, spearman_r = {}, spearman_p = {}, mse = {}, r2 = {}'.format(*new_row))
            f.write('\n')
    

    def set_best_epoch(self, epoch, val_loss, val_pearson, val_r2):
        '''
        Update the best validation loss.
        '''
        self.__best_val_loss = val_loss
        self.__best_val_pearson = val_pearson
        self.__best_val_r2 = val_r2
        self.__best_epoch = epoch


    def reset_best_epoch(self):
        '''
        Reset the best epoch (when starting a new round of training).
        '''
        self.__best_val_loss = 100.0
        self.__best_val_pearson = 0
        self.__best_val_r2 = 0
        self.__best_epoch = 0


    def get_best_epoch(self):
        '''
        Return the best epoch and its corresponding metrics (for early stopping).
        '''
        return self.__best_epoch, self.__best_val_pearson, self.__best_val_r2, self.__best_val_loss


    def get_path_to_model(self, fold, seed):
        '''
        Return the path to the model with the selected parameters.
        '''
        return os.path.join(self.__models_dir, '{}.{}.pt'.format(fold, seed))
    
    
    def signal_completion(self):
        '''
        Record the finish time.
        '''
        current_dt = datetime.now()
        dt = current_dt.strftime('%y-%m-%d %H:%M:%S')
        if self.__resuming == False:
            runtime = current_dt - self.__date_time
            # record in log file
            with open(self.__log_file, 'a') as f:
                f.write('\n{} -- RUN COMPLETE'.format(dt))
                f.write('\n\ttotal runtime {}'.format(runtime))
                f.write('\n')
        else:
            with open(self.__log_file, 'a') as f:
                f.write('\n{} -- RUN COMPLETE'.format(dt))
                f.write('\n\tcannot calculate total runtime due to interruption')
                f.write('\n')


    def __signal_resumption(self):
        '''
        Record the resumption time and current fold information.
        '''
        current_dt = datetime.now()
        dt = current_dt.strftime('%y-%m-%d %H:%M:%S')
        # record in log file
        with open(self.__log_file, 'a') as f:
            f.write('\n--!--!--!-- INTERRUPTION --!--!--!--')
            f.write('\n------------------------------------\n')
            f.write('\n{} -- RESUMING RUN AFTER PREVIOUS:'.format(dt))
            f.write('\n\tfold = {}, seed = {}'.format(self.__current_fold_no, self.__current_seed_no))
            f.write('\n')
