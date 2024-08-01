import torch
import torch.nn as nn
from seq2exp_functions import calculate_pooling_output_length, calculate_conv_output_length


class AugmentedSeq2Exp(nn.Module): 
    '''
    Based on the Xpresso model (Agarwal et al.)
    '''
    def __init__(self, dropout = 0.5, input_length = 2000, atac_channels = True, dna_channels = True, include_disp = False, include_sigma = False):
        super(AugmentedSeq2Exp, self).__init__()

        self.__input_length = input_length
        self.__atac_channel = atac_channels
        self.__dna_channels = dna_channels

        input_channels = 0
        if self.__atac_channel == True:
            input_channels += 1
        if self.__dna_channels == True:
            input_channels += 4

        conv_1_kernel_size = 6
        conv_1_out_channels = 128
        self.conv_1 = nn.Conv1d(input_channels, conv_1_out_channels, kernel_size = conv_1_kernel_size, dilation = 1, padding = 0)
        conv_1_out_length = calculate_conv_output_length(input_length, 0, conv_1_kernel_size, 1, 1)

        pool_1_kernel_size = 8
        pool_1_stride = pool_1_kernel_size
        self.pool_1 = nn.MaxPool1d(pool_1_kernel_size, stride = pool_1_stride, padding = 0)
        pool_1_out_length = calculate_pooling_output_length(conv_1_out_length, 0, pool_1_kernel_size, pool_1_stride)

        conv_2_kernel_size = 9
        conv_2_out_channels = 64 # 32
        self.conv_2 = nn.Conv1d(conv_1_out_channels, conv_2_out_channels, kernel_size = conv_2_kernel_size, dilation = 2, padding = 0)
        conv_2_out_length = calculate_conv_output_length(pool_1_out_length, 0, conv_2_kernel_size, 1, 2)
        
        pool_2_kernel_size = 8
        pool_2_stride = pool_2_kernel_size
        self.pool_2 = nn.MaxPool1d(pool_2_kernel_size, stride = pool_2_stride, padding = 0)
        pool_2_out_length = calculate_pooling_output_length(conv_2_out_length, 0, pool_2_kernel_size, pool_2_stride)

        self.flat = nn.Flatten()

        lin_1_in_length = pool_2_out_length * conv_2_out_channels
        lin_1_out_length = lin_1_in_length
        lin_2_in_length = lin_1_out_length

        self.lin_1 = nn.Linear(lin_1_in_length, lin_1_out_length)
        self.lin_2 = nn.Linear(lin_2_in_length, 64)
        self.output = nn.Linear(64, 1)

        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(p = dropout)

    def forward(self, promoter):

        x0 = self.dropout(promoter)
        x1 = self.pool_1(self.conv_1(x0))
        x2 = self.pool_2(self.conv_2(x1))

        x3 = self.dropout(self.flat(x2))
        x4 = self.dropout(self.activation(self.lin_1(x3)))
        x5 = self.dropout(self.activation(self.lin_2(x4)))

        return self.output(x5)

    def get_channels(self):
        return { 'atac': self.__atac_channel, 'dna': self.__dna_channels }



class AugmentedSeq2ExpSequential(nn.Module): 
    '''
    Sequential version of AugmentedSeq2Exp for explainers.
    '''
    def __init__(self, dropout = 0.5, input_length = 2000, atac_channel = True, dna_channels = True):
        super(AugmentedSeq2ExpSequential, self).__init__()

        self.__input_length = input_length
        self.__atac_channel = atac_channel
        self.__dna_channels = dna_channels

        input_channels = 0
        if atac_channel == True:
            input_channels += 1
        if dna_channels == True:
            input_channels += 4

        conv_1_kernel_size = 6
        conv_1_out_channels = 128
        conv_1_out_length = calculate_conv_output_length(input_length, 0, conv_1_kernel_size, 1, 1)

        pool_1_kernel_size = 8
        pool_1_stride = pool_1_kernel_size
        pool_1_out_length = calculate_pooling_output_length(conv_1_out_length, 0, pool_1_kernel_size, pool_1_stride)

        conv_2_kernel_size = 9
        conv_2_out_channels = 64 # 32
        conv_2_out_length = calculate_conv_output_length(pool_1_out_length, 0, conv_2_kernel_size, 1, 2)
        
        pool_2_kernel_size = 8
        pool_2_stride = pool_2_kernel_size
        pool_2_out_length = calculate_pooling_output_length(conv_2_out_length, 0, pool_2_kernel_size, pool_2_stride)

        lin_1_in_length = pool_2_out_length * conv_2_out_channels
        lin_1_out_length = lin_1_in_length
        lin_2_in_length = lin_1_out_length

        self.conv_layers = nn.Sequential(nn.Dropout(p = dropout),
                                         nn.Conv1d(input_channels, conv_1_out_channels, kernel_size = conv_1_kernel_size, dilation = 1, padding = 0),
                                         nn.MaxPool1d(pool_1_kernel_size, stride = pool_1_stride, padding = 0),
                                         nn.Conv1d(conv_1_out_channels, conv_2_out_channels, kernel_size = conv_2_kernel_size, dilation = 2, padding = 0),
                                         nn.MaxPool1d(pool_2_kernel_size, stride = pool_2_stride, padding = 0),
                                         nn.Flatten())
        
        self.dense_layers = nn.Sequential(nn.Dropout(p = dropout),
                                          nn.Linear(lin_1_in_length, lin_1_out_length),
                                          nn.ReLU(),
                                          nn.Dropout(p = dropout),
                                          nn.Linear(lin_2_in_length, 64),
                                          nn.ReLU(),
                                          nn.Dropout(p = dropout),
                                          nn.Linear(64, 1))

    def forward(self, promoter):
        x = self.conv_layers(promoter)
        x = self.dense_layers(x)
        return x

    def get_channels(self):
        return { 'atac': self.__atac_channel, 'dna': self.__dna_channels }

