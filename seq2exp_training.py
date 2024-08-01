import numpy as np
import sys, gc
import torch
import torch.nn as nn
from tqdm import tqdm
import models
from multiome import MultiOmicDataset
from seq2exp_functions import calculate_regressor_metrics



def make_dataloaders(train_indices, val_indices, data_dir, batch_size = 512, num_workers = 1):
    '''
    Prepare direct dataloaders for MultiOmicDataset.
    '''
    data = MultiOmicDataset(data_dir)
    promoters, atac, rna = data.fetch_samples(train_indices)
    train_data = torch.utils.data.TensorDataset(promoters.float(), atac.float(), rna.float())
    del promoters, atac, rna
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    del train_data
    promoters, atac, rna = data.fetch_samples(val_indices)
    val_data = torch.utils.data.TensorDataset(promoters.float(), atac.float(), rna.float())
    del promoters, atac, rna
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    del val_data
    del data
    gc.collect()
    return train_dataloader, val_dataloader


def train_regressor_with_early_stop(model, train_dl, val_dl, logger, opt, lr, wd, max_epochs, val_interval, device, loss_fn = nn.MSELoss()):
    '''
    Train a model using given parameters, and use the validation set for early stopping.
    '''
    if opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, weight_decay = wd)
    elif opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = wd)
    logger.reset_best_epoch()
    logger.reset_loss_curves()
    # find the number of arguments required for the model
    input_channels = model.get_channels()
    # start training loop
    print('Training...')
    logger.record_training_start()
    for epoch in tqdm(range(max_epochs), total = max_epochs):
        epoch_loss = 0.0  # for one full epoch
        model.train()
        if epoch == max_epochs - 1:
            with torch.no_grad():
                true_exp = torch.empty(0).to(device)
                pred_exp_probs = torch.empty(0).to(device)
        for promoter_data, atac_data, rna_data in train_dl:
            promoter_data, atac_data, rna_data = promoter_data.to(device), atac_data.to(device), rna_data.to(device)
            optimizer.zero_grad()
            if input_channels['atac'] == True:
                if len(atac_data.shape) == 2:
                    ready_atac = atac_data.reshape((atac_data.shape[0], -1, atac_data.shape[1]))
                else:
                    ready_atac = atac_data
            if input_channels['dna'] == True and input_channels['atac'] == False:
                output = model(promoter_data)
            elif input_channels['dna'] == True and input_channels['atac'] == True:
                promoter_plus = torch.cat([promoter_data, ready_atac], dim = 1)
                output = model(promoter_plus)
            elif input_channels['dna'] == False and input_channels['atac'] == True:
                output = model(ready_atac)
            else:
                print('Check input channels!')
                sys.exit(2)
            train_loss = loss_fn(output.flatten(), rna_data.flatten())
            train_loss.backward()
            optimizer.step()
            epoch_loss += train_loss.item()
            if epoch == max_epochs - 1:
                with torch.no_grad():
                    true_exp = torch.cat((true_exp, rna_data.detach()), 0)
                    pred_exp_probs = torch.cat((pred_exp_probs, output.detach()), 0)
        if epoch == max_epochs - 1:
            if len(true_exp.shape) == 1:
                train_metrics = calculate_regressor_metrics(pred_exp_probs.cpu().flatten(), true_exp.cpu())
            else:
                r2s = []
                rhos = []
                for task in range(true_exp.shape[1]):
                    metrics = calculate_regressor_metrics(pred_exp_probs.cpu()[:, task], true_exp.cpu()[:, task])
                    r2s.append(metrics['r2'])
                    rhos.append(metrics['pearson_r'])
                train_metrics = {}
                train_metrics['r2'] = np.mean(r2s)
                train_metrics['pearson_r'] = np.mean(rhos)
        # log training loss for epoch
        logger.append_to_train_loss(epoch, epoch_loss / len(train_dl))
        # validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                true_exp = torch.empty(0).to(device)
                pred_exp_probs = torch.empty(0).to(device)
                for promoter_data, atac_data, rna_data in val_dl:
                    promoter_data, atac_data, rna_data = promoter_data.to(device), atac_data.to(device), rna_data.to(device)
                    if input_channels['atac'] == True:
                        if len(atac_data.shape) == 2:
                            ready_atac = atac_data.reshape((atac_data.shape[0], -1, atac_data.shape[1]))
                        else:
                            ready_atac = atac_data
                    if input_channels['dna'] == True and input_channels['atac'] == False:
                        output = model(promoter_data)
                    elif input_channels['dna'] == True and input_channels['atac'] == True:
                        promoter_plus = torch.cat([promoter_data, ready_atac], dim = 1)
                        output = model(promoter_plus)
                    elif input_channels['dna'] == False and input_channels['atac'] == True:
                        output = model(ready_atac)
                    else:
                        print('Check input channels!')
                        sys.exit(2)
                    loss = loss_fn(output.flatten(), rna_data.flatten())
                    val_loss += loss.item()
                    true_exp = torch.cat((true_exp, rna_data), 0)
                    pred_exp_probs = torch.cat((pred_exp_probs, output), 0)
                current_val_loss = val_loss / len(val_dl)
                logger.append_to_val_loss(epoch, current_val_loss, save_to_log = True)
                # if the validation loss improved, save a copy of the model parameters
                best_val_loss = logger.get_best_epoch()[-1]
                if current_val_loss < best_val_loss:
                    if len(true_exp.shape) == 1:
                        metrics = calculate_regressor_metrics(pred_exp_probs.cpu().flatten(), true_exp.cpu())
                        val_r2 = metrics['r2']
                        val_rho = metrics['pearson_r']
                    else:
                        r2s = []
                        rhos = []
                        for task in range(true_exp.shape[1]):
                            metrics = calculate_regressor_metrics(pred_exp_probs.cpu()[:, task], true_exp.cpu()[:, task])
                            r2s.append(metrics['r2'])
                            rhos.append(metrics['pearson_r'])
                        val_r2 = np.mean(r2s)
                        val_rho = np.mean(rhos)
                    logger.set_best_epoch(epoch, current_val_loss, val_rho, val_r2)
                    torch.save(model.state_dict(), logger.get_model_save_path())
    # save loss curves
    logger.record_final_train_metrics(train_metrics['pearson_r'], train_metrics['r2'])
    logger.save_loss_curve_data()
    logger.save_val_results()


def load_best_model(model_name, outer_fold, logger):
    '''
    Load model using the best parameters saved to file.
    '''
    outer_fold, trial, med_inner_fold = logger.get_best_params(outer_fold)
    model_class = getattr(models, model_name)
    model = model_class()
    model.load_state_dict(torch.load(logger.get_path_to_model(outer_fold, trial, med_inner_fold)))
    return model
    

def test_regressor(model, test_dl, logger, device = torch.device('cpu'), batch_size = 512, num_workers = 1):
    '''
    Test a trained model.
    '''
    print('Testing...')
    model.to(device)
    model.eval()
    true_exp = torch.empty(0).to(device)
    pred_exp_probs = torch.empty(0).to(device)
    # find the number of arguments required for the model
    input_channels = model.get_channels()
    with torch.no_grad():
        for promoter_data, atac_data, rna_data in test_dl:
            promoter_data, atac_data, rna_data = promoter_data.to(device), atac_data.to(device), rna_data.to(device)
            if input_channels['atac'] == True:
                if len(atac_data.shape) == 2:
                    ready_atac = atac_data.reshape((atac_data.shape[0], -1, atac_data.shape[1]))
                else:
                    ready_atac = atac_data
            if input_channels['dna'] == True and input_channels['atac'] == False:
                output = model(promoter_data)
            elif input_channels['dna'] == True and input_channels['atac'] == True:
                promoter_plus = torch.cat([promoter_data, ready_atac], dim = 1)
                output = model(promoter_plus)
            elif input_channels['dna'] == False and input_channels['atac'] == True:
                output = model(ready_atac)
            else:
                print('Check input channels!')
                sys.exit(2)
            true_exp = torch.cat((true_exp, rna_data), 0)
            pred_exp_probs = torch.cat((pred_exp_probs, output), 0)
        # log results
        if len(true_exp.shape) == 1:
            metrics = calculate_regressor_metrics(pred_exp_probs.cpu().flatten(), true_exp.cpu())
        else:
            metric_collection = { 'pearson_r': [], 'pearson_p': [], 'spearman_r': [], 'spearman_p': [], 'mse': [], 'r2': [] }
            for task in range(true_exp.shape[1]):
                task_metrics = calculate_regressor_metrics(pred_exp_probs.cpu()[:, task], true_exp.cpu()[:, task])
                for m in metric_collection.keys():
                    metric_collection[m].append(task_metrics[m])
            metrics = { m: np.mean(metric_collection[m]) for m in metric_collection.keys() }
        logger.save_test_metrics(metrics)


def regressor_prediction(model, test_dl, device = torch.device('cpu'), batch_size = 512, num_workers = 1):
    '''
    Make predictions using a trained model.
    '''
    model.to(device)
    model.eval()
    true_exp = torch.empty(0).to(device)
    pred_exp = torch.empty(0).to(device)
    # find the number of arguments required for the model
    input_channels = model.get_channels()
    with torch.no_grad():
        for promoter_data, atac_data, rna_data in test_dl:
            promoter_data, atac_data, rna_data = promoter_data.to(device), atac_data.to(device), rna_data.to(device)
            if input_channels['atac'] == True:
                if len(atac_data.shape) == 2:
                    ready_atac = atac_data.reshape((atac_data.shape[0], -1, atac_data.shape[1]))
                else:
                    ready_atac = atac_data
            if input_channels['dna'] == True and input_channels['atac'] == False:
                output = model(promoter_data)
            elif input_channels['dna'] == True and input_channels['atac'] == True:
                promoter_plus = torch.cat([promoter_data, ready_atac], dim = 1)
                output = model(promoter_plus)
            elif input_channels['dna'] == False and input_channels['atac'] == True:
                output = model(ready_atac)
            else:
                print('Check input channels!')
                sys.exit(2)
            true_exp = torch.cat((true_exp, rna_data), 0)
            pred_exp = torch.cat((pred_exp, output), 0)
    return true_exp.cpu(), pred_exp.cpu()
