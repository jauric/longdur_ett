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
            y_predictions_batch_flat_scaled = y_predictions_batch_scaled.cpu().numpy().reshape(-1,1)
            y_targets_batch_flat_scaled     = y_targets_batch_scaled.numpy().reshape(-1,1)

            # Rescale
            y_predictions_batch_flat = scaler_OT.inverse_transform(y_predictions_batch_flat_scaled)
            y_targets_batch_flat     = scaler_OT.inverse_transform(y_targets_batch_flat_scaled)

            # Reshape back from flattened [batch_size x forecast_length, 1] --> [batch_size, forecast_length]
            forecast_length = y_predictions_batch_scaled.shape[1]

            y_predictions_batch = y_predictions_batch_flat.reshape(-1, forecast_length)
            y_targets_batch     = y_targets_batch_flat.reshape(-1, forecast_length)

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



    return y_predictions, y_targets, test_mse, test_mae


def plot_test_results(
        prepared_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict],
        test_results: tuple[np.ndarray, np.ndarray, float, float], 
        context_length: int, 
        forecast_length: int,
        ):

    # Unpack concatenated batches
    # Size: [batch_size x test_batch_qty, forecast_length] = [test_sample_qty, forecast_length]
    _, _, _, _, training_split, validation_split, test_split = prepared_data 
    y_predictions, y_targets, _, _ = test_results     

    test_sample_qty                = y_predictions.shape[0]


    ################## MEAN SQUARED ERROR ##################

    # Calculate mse for every test sample, determine min(), max(), mean() values
    mse = [mean_squared_error(y_targets[n], y_predictions[n]) for n in range(test_sample_qty)]

    mse = np.array(mse)     # convert to np.array

    mse_min  = mse.min()
    mse_max  = mse.max()
    mse_mean = mse.mean()

    mse_mean_plotting = [mse_mean for k in range(test_sample_qty)]   # to plot a line at the mean

    # Determine the index k of the (first) samples with min() and max() values
    k_mse_min = mse.argmin()
    k_mse_max = mse.argmax()

    # Plot mse over sample range
    plt.figure(figsize=(12, 5))
    plt.plot(range(y_predictions.shape[0]), mse, linewidth=1, color="blue", label="MSE(k)")
    plt.plot(range(y_predictions.shape[0]), mse_mean_plotting, linewidth=2, color="orange", label=f"MSE_mean = {mse_mean:.4f}")
    plt.scatter(k_mse_max, mse_max, s = 75, color="red", marker="^", label=f"MSE_max = {mse_max:.4}, k = {k_mse_max}" )
    plt.scatter(k_mse_min, mse_min, s = 75, color="red", marker="v", label=f"MSE_min = {mse_min:.4}, k = {k_mse_min}")
    plt.xlabel("Test Sample Index k")
    plt.ylabel('MSE')
    plt.title("Mean Squared Error")
    plt.legend()
    plt.show()


    ################## MEAN ABSOLUTE ERROR ##################

    # Calculate mae for every test sample, determine min(), max(), mean() values
    mae = [mean_absolute_error(y_targets[n], y_predictions[n]) for n in range(test_sample_qty)]

    mae = np.array(mae)     # convert to np.array

    mae_min  = mae.min()
    mae_max  = mae.max()
    mae_mean = mae.mean()

    mae_mean_plotting = [mae_mean for k in range(test_sample_qty)]   # to plot a line at the mean

    # Determine the index k of the (first) samples with min() and max() values
    k_mae_min = mae.argmin()
    k_mae_max = mae.argmax()

    # Plot mae over sample range
    plt.figure(figsize=(12, 5))
    plt.plot(range(y_predictions.shape[0]), mae, linewidth=1, color="blue", label="MAE(k)")
    plt.plot(range(y_predictions.shape[0]), mae_mean_plotting, linewidth=2, color="orange", label=f"MAE_mean = {mae_mean:.4f}")
    plt.scatter(k_mae_max, mae_max, s = 75, color="red", marker="^", label=f"MAE_max = {mae_max:.4}, k = {k_mae_max}" )
    plt.scatter(k_mae_min, mae_min, s = 75, color="red", marker="v", label=f"MAE_min = {mae_min:.4}, k = {k_mae_min}")
    plt.xlabel("Test Sample Index k")
    plt.ylabel("MAE")
    plt.title("Mean Absolute Error")
    plt.legend()
    plt.show()     


    ################## LINE PLOT, FIRST SAMPLE (k = 0) ##################

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
        y_targets[0],
        linewidth=2,
        label="y_target",
        color="tab:blue",
    )

    # Plot predictions
    plt.plot(
        time[context_length:(context_length+forecast_length)],
        y_predictions[0],
        linewidth=2,
        label=f"y_prediction, MSE_0 ={mse[0]:.4f} ",
        color="tab:orange",
    )

    plt.xlabel("Time (h)")
    plt.ylabel("OT")
    plt.title("First Prediction, Sample k = 0")

    plt.legend()
    plt.tight_layout()
    plt.show()


    ################## LINE PLOT, MSE_MIN ##################

    # Create time axis, use the input and prediction window as bounds
    time = np.arange(-context_length, forecast_length, 1)

    # Plot the results
    plt.figure(figsize=(12, 5))

    # Plot inputs
    plt.plot(
        time[:context_length],
        test_split[(k_mse_min-context_length):k_mse_min]["OT"],
        linewidth=2,
        label="x_inputs",
        color="navy",
    )

    # Plot targets
    plt.plot(
        time[context_length:(context_length+forecast_length)],
        y_targets[k_mse_min],
        linewidth=2,
        label="y_target",
        color="tab:blue",
    )

    # Plot predictions
    plt.plot(
        time[context_length:(context_length+forecast_length)],
        y_predictions[k_mse_min],
        linewidth=2,
        label=f"y_prediction, MSE_min ={mse_min:.4f} ",
        color="tab:orange",
    )

    plt.xlabel("Time (h)")
    plt.ylabel("OT")
    plt.title("Best Prediction, Sample k_mse_min")

    plt.legend()
    plt.tight_layout()
    plt.show()


    ################## LINE PLOT, MSE_MAX ##################

    # Create time axis, use the input and prediction window as bounds
    time = np.arange(-context_length, forecast_length, 1)

    # Plot the results
    plt.figure(figsize=(12, 5))

    # Plot inputs
    plt.plot(
        time[:context_length],
        test_split[(k_mse_max-context_length):k_mse_max]["OT"],
        linewidth=2,
        label="x_inputs",
        color="navy",
    )

    # Plot targets
    plt.plot(
        time[context_length:(context_length+forecast_length)],
        y_targets[k_mse_max],
        linewidth=2,
        label="y_target",
        color="tab:blue",
    )

    # Plot predictions
    plt.plot(
        time[context_length:(context_length+forecast_length)],
        y_predictions[k_mse_max],
        linewidth=2,
        label=f"y_prediction, MSE_max ={mse_max:.4f} ",
        color="tab:orange",
    )

    plt.xlabel("Time (h)")
    plt.ylabel("OT")
    plt.title("Worst Prediction, Sample k_max_min")

    plt.legend()
    plt.tight_layout()
    plt.show()