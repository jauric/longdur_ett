import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error




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
    y_test_targets_scaled = []
    y_test_predictions_scaled = []

    with torch.no_grad():
        # no gradient calculation required

        for x_test_batch, y_test_targets_batch in test_batches:

            x_test_batch        = x_test_batch.to(device)

            # Make the prediction
            y_test_predictions_batch = model(x_test_batch)

            # Collect all y_prediction and y_target batches simultaneously (also works whenthe batches are shuffled)
            # Convert tensors to np.ndarrays for analysis, must be done on cpu so .cpu() on the predictions. 
            # Targets are on cpu already, never sent to device
            y_test_predictions_scaled.append(y_test_predictions_batch.cpu().numpy())
            y_test_targets_scaled.append(y_test_targets_batch.numpy())


    # Transform the collection of np.arrays to a single array for error calculation
    y_test_predictions_scaled = np.concatenate(y_test_predictions_scaled, axis=0)
    y_test_targets_scaled = np.concatenate(y_test_targets_scaled, axis=0)

    # Select the right scaler for the OT signal
    scaler_OT = scalers_dict["OT"]

    # Convert predictions back to source scale
    y_test_predictions = scaler_OT.inverse_transform(y_test_predictions_scaled)
    y_test_targets = scaler_OT.inverse_transform(y_test_targets_scaled)

    ### Calculating metrics
    test_mse = mean_squared_error(y_test_targets, y_test_predictions)
    test_mae = mean_absolute_error(y_test_targets, y_test_predictions)

    print(f"\nTest MSE: {test_mse:.8f}")
    print(f"Test MAE: {test_mae:.8f}")

    return y_test_predictions, y_test_targets, test_mse, test_mae