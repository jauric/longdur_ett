import pandas as pd
import numpy as np
from copy import deepcopy



def data_import(url: str) -> pd.DataFrame:

    data = pd.read_csv(url, sep = ",")
    data.set_index('date', inplace=True)

    return data



def data_prep(
        source_data: pd.DataFrame, 
        train_length_months: int, 
        vali_length_months: int, 
        test_length_months: int, 
        context_length: int,
        scaler: object        
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    '''
    Converting the source DataFrame into split and scaled datasets, while returning the scalers.
    Currently only for ETTh data with 1h intervals, can be extended to cover ETTm1 with 15min steps.
    
    Args:
        scaler: enter a scaler object from e.g. sklearn.preprocessing, incl. freature_ranges if required.
    '''

    ############### Creating raw data splits

    # Converting months into hours to match the source data
    hours_per_month   = 30 * 24     # hours

    training_length   = train_length_months * hours_per_month         # hours
    validation_length = vali_length_months * hours_per_month          # hours
    test_length       = test_length_months * hours_per_month          # hours

    # Create training, validation, and test splits
    training_split    = source_data.iloc[:training_length]
    validation_split  = source_data.iloc[training_length:(training_length + validation_length)]
    test_split        = source_data.iloc[(training_length + validation_length):(training_length + validation_length + test_length)]
    discarded_split   = source_data.iloc[(training_length + validation_length + test_length):]

    timesteps = training_split.shape[0] + validation_split.shape[0] + test_split.shape[0]
    timesteps_source = source_data.shape[0]         # Number of rows in the source data set

    print("Timesteps in the source dataset: ", timesteps_source, "hours")
    print("Timesteps in the truncated split: ", timesteps, "hours")
    print("Timesteps in the training split: ", training_split.shape[0], "hours")
    print("Timesteps in the discarded split: ", discarded_split.shape[0], "hours")
    print("The training/validation/test split relative to the truncated set [%]: ", 
          training_split.shape[0]/timesteps*100, "/", validation_split.shape[0]/timesteps*100, "/", test_split.shape[0]/timesteps*100)


    ############### Assembling the actual data sets from the splits, considering moving window sampling

    # No changes needed
    training_set = training_split

    # Extending the validation split, x_val should start at -context in the training set, so first val_prediction is the first unseen entry in the validation split (moving window, see Dataset __getitem__!)
    validation_set = pd.concat([ training_split.iloc[ -context_length : ], validation_split])

    # Extending the testing split, respectively
    test_set = pd.concat([ validation_split.iloc[ -context_length : ], test_split])


    ############### Scaling the data

    # Set up empty DataFrames to hold the scaled data with the right index and columns; 0 = training_set, 1 = validation_set, 2 = test_set
    training_set_scaled   = pd.DataFrame(index = training_set.index, columns = training_set.columns)
    validation_set_scaled = pd.DataFrame(index = validation_set.index, columns = validation_set.columns)
    test_set_scaled       = pd.DataFrame(index = test_set.index, columns = test_set.columns)

    # Initialize a dictionary to store the specific scaler for each channel
    scalers_dict = {}

    for column in training_set.columns:

        # New scaler object per column
        col_scaler = deepcopy(scaler)  

        # Fit the scaler on the training set of each signal only
        col_scaler.fit(training_set[[column]])

        # Save this column's scaler for later inverse transformation of predictions
        scalers_dict[column] = col_scaler

        # Transform the training, validation, and test sets of each signal using the fitted scaler. All set have the same index and columns as the original data frames.
        training_set_scaled[column] = col_scaler.transform(training_set[[column]])
        validation_set_scaled[column] = col_scaler.transform(validation_set[[column]])
        test_set_scaled[column] = col_scaler.transform(test_set[[column]])


    return training_set_scaled, validation_set_scaled, test_set_scaled, scalers_dict, training_split, validation_split, test_split
    # each set and split has shape [len, 7]