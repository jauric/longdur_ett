import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

import pandas as pd


def model_test(
    model: object,
    best_state_dict: dict,
    data_batches: tuple[DataLoader, DataLoader, DataLoader], 
    scalers_dict: dict,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:

    # Unpacking the batches
    test_batches   = data_batches[2]

    # Select device for computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load the optimized model states, send model for execution on defined device, start evaluation
    model.load_state_dict(best_state_dict)
    model.to(device)
    model.eval()

    # Initialize arrays for bookkeeping
    y_targets = []
    y_predictions = []

    # Select the right scaler for the OT signal
    scaler_OT = scalers_dict["OT"]

    with torch.no_grad():
        # no gradient calculation required

        for x_batch, y_targets_batch_scaled in test_batches:

            x_batch = x_batch.to(device)       

            # Make the prediction
            y_predictions_batch_scaled = model(x_batch)      # shape = [batch_size, forecast_length]

            # Flatten to 1 dimension for inverse scaling, inverse_transform() needs input of [timesteps, n_features=1] because it was fitted on 1 feature only
            # Reshape [batch_size, forecast_length] --> [batch_size x forecast_length, 1]
            # Convert tensors to np.ndarrays for analysis, must be done on cpu so .cpu() on the predictions before .numpy(). 
            # Targets are on cpu already, never sent to device
            y_predictions_batch_scaled.cpu().numpy().reshape(-1,1)
            y_targets_batch_scaled.numpy().reshape(-1,1)

            # Rescale
            y_predictions_batch = scaler_OT.inverse_transform(y_predictions_batch_scaled)
            y_targets_batch = scaler_OT.inverse_transform(y_targets_batch_scaled)

            # Reshape back from flattened [batch_size x forecast_length, 1] --> [batch_size, forecast_length]
            forecast_length = y_predictions_batch.shape[1]
            y_predictions_batch.reshape(-1, forecast_length)
            y_targets_batch.reshape(-1, forecast_length)

            # Collect all y_prediction and y_target batches simultaneously (also works whenthe batches are shuffled)
            # List shape: [batch_qty], batches sit side by side with samples stacked within
            y_predictions.append(y_predictions_batch)
            y_targets.append(y_targets_batch)


    # Transform the list of np.arrays to a single array for error calculation   shape: [batch_size x batch_qty = sample_qty, forecast_horizon]
    # All samples of all batches are stacked on top of each other (axis = 0 --> stacking in first dim of each batch = sample index)
    y_predictions = np.concatenate(y_predictions, axis=0)
    y_targets     = np.concatenate(y_targets, axis=0)


    ### Calculating metrics
    test_mse = mean_squared_error(y_targets, y_predictions)
    test_mae = mean_absolute_error(y_targets, y_predictions)

    print(f"\nTest MSE: {test_mse:.8f}")
    print(f"Test MAE: {test_mae:.8f}")



    return y_predictions, y_targets, test_mse, test_mae, y_predictions_batch_scaled, y_targets_batch_scaled     # remove last two after test



def prediction_plot(
        prepared_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict],
        test_results: tuple[np.ndarray, np.ndarray, float, float], 
        context_length: int, 
        forecast_length: int):

    # Unpack time series to plot
    _, _, _, _, training_split, validation_split, test_split = prepared_data 
    y_test_predictions, y_test_targets, _, _, _, _           = test_results     # remove last two after test

    # Create time axis, use the input and prediction window as bounds
    time = np.arange(-context_length, forecast_length, 1)

    # Plot the results
    plt.figure(figsize=(12, 5))

    # Plot inputs
    plt.plot(
        time[:context_length],
        validation_split[-context_length:]["OT"],
        linewidth=2,
        label="x_inputs",
        color="navy",
    )

    # Plot targets
    plt.plot(
        time[context_length:(context_length+forecast_length)],
        y_test_targets[0],
        linewidth=2,
        label="y_target",
        color="tab:blue",
    )

    # Plot predictions
    plt.plot(
        time[context_length:(context_length+forecast_length)],
        y_test_predictions[0],
        linewidth=2,
        label="y_predicction",
        color="tab:orange",
    )

    plt.xlabel("Time (h)")
    plt.ylabel("OT")

    plt.legend()
    plt.tight_layout()
    plt.show()