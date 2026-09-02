import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader



class Sample(Dataset):

    '''
    Sampling for multivariate-to-univariate(OT) one-shot forecasting.
    '''

    def __init__(self, data: pd.DataFrame, context_length: int, forecast_length: int):

        # Converting DataFrame to 32-bit float torch.tensor for better processing, going through a np.array for compatibility reasons
        self.feature_data    = torch.from_numpy(data.to_numpy(dtype=np.float32))          # input x -- either the whole training, validation or testing split
        self.target_data     = torch.from_numpy(data["OT"].to_numpy(dtype=np.float32))    # prediction target y = whole OT time series

        self.context_length  = context_length       # L 
        self.forecast_length = forecast_length      # H 


    def __len__(self):

        # Total number of samples in the dataset
        sample_qty = (self.feature_data.shape[0] - self.context_length - self.forecast_length) + 1      # N = ((T-L-H)/S)+1     | Stride S = 1 timestep

        return sample_qty


    def __getitem__(self, sample_index):
        # Construct the sample from the input data

        # x = sample input data = the full multivariate time series but restricted to the context length = only ranging from index to (index + context_length) (index = 0,1...)
        x_inputs_array = self.feature_data[sample_index : (sample_index + self.context_length)]           # last value of that range is EXCLUDED, the first is INCLUDED

        # y = data of the correlated target output -- here: univariate output of OT, but positioned behind the the training window and being as wide as the forecast window
        y_targets_array = self.target_data[(sample_index + self.context_length) : (sample_index + self.context_length + self.forecast_length)]     # MIND! last value of that range is EXCLUDED, the first is INCLUDED

        return x_inputs_array, y_targets_array




def create_batches(
        prepared_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict], 
        context_length: int, 
        forecast_length: int,
        batch_size: int, 
        training_shuffle: bool
        ) -> tuple[DataLoader, DataLoader, DataLoader]:

    '''
    Convert split and scaled data into PyTorch DataLoaders.

    Args:
        prepared_data: the output of function in data_utils.py: data_prep() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]

    Returns:
        out: tuple of the batched samples for training, validation, and test; each with x_inputs and y_targets
    '''
    

    # Initialize data sampling
    training_dataset = Sample(
        data            = prepared_data[0],
        context_length  = context_length,
        forecast_length = forecast_length
    )

    validation_dataset = Sample(
        data            = prepared_data[1],
        context_length  = context_length,
        forecast_length = forecast_length
    )

    test_dataset = Sample(
        data            = prepared_data[2],
        context_length  = context_length,
        forecast_length = forecast_length
    )

    # Load samples and initialize batches
    training_batches = DataLoader(
        dataset    = training_dataset,
        batch_size = batch_size,
        shuffle    = training_shuffle
    )

    validation_batches = DataLoader(
        dataset    = validation_dataset,
        batch_size = batch_size,
        shuffle    = False
    )

    test_batches = DataLoader(
        dataset    = test_dataset,
        batch_size = batch_size,
        shuffle    = False
    )

    # Shape check: extract batches for x_inputs and y_targets data
    x_inputs_batch, y_targets_batch = next(iter(training_batches))

    print("Input shape:", x_inputs_batch.shape)
    print("Target shape:", y_targets_batch.shape)
    print("Number of batches:", len(training_batches))


    return training_batches, validation_batches, test_batches
    # each batch = x_inputs, y_targets, with 
    # x_inputs.shape  = [batch_size, len = context_length, channels = 7], 
    # y_targets.shape = [batch_size, len = forecast_length] -> univariate, no channel dimension