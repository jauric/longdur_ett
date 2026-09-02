import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def model_training(
        model: object, 
        data_batches: tuple[DataLoader, DataLoader, DataLoader], 
        max_epochs: int, 
        patience_epochs: int,
        optimizer_learningrate: float
        ) -> dict :

    '''
    Takes the model architecture, training/validation data batches, main training parameters, and returns the
    best model states after early stopping conditions were reached.
    '''

    ############### Unpacking the batches
    training_batches   = data_batches[0]
    validation_batches = data_batches[1]


    ############### Training auxiliaries

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = optimizer_learningrate)

    # Print number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} trainable parameters.")

    # Select device for computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Execute model on defined device
    model.to(device)


    ############### Training loop
    
    training_losses = []
    validation_losses = []

    # Early-stopping bookkeeping (initialize once, outside the epoch loop to avoid constant overwriting)
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    best_state_dict = None
    best_epoch = None


    for epoch in range(max_epochs):

        # Training Step
        model.train()
        epoch_training_loss_sum = 0.0

        for x_training_batch, y_training_targets_batch in training_batches:

            x_training_batch         = x_training_batch.to(device)
            y_training_targets_batch = y_training_targets_batch.to(device)

            optimizer.zero_grad()

            y_training_predictions_batch = model(x_training_batch)

            # Calculate the training loss
            batch_training_loss = criterion(
                y_training_predictions_batch,
                y_training_targets_batch,
            )

            batch_training_loss.backward()
            optimizer.step()

            epoch_training_loss_sum += batch_training_loss.item()

        epoch_training_loss_avg = epoch_training_loss_sum / len(training_batches)     # average epoch loss per batch
        training_losses.append(epoch_training_loss_avg)


        # Evaluation step
        model.eval()
        epoch_validation_loss_sum = 0.0

        with torch.no_grad():
        # disables gradient calculation, no gradients for backpropagation needed here

            for x_validation_batch, y_validation_targets_batch in validation_batches:
                # Note: less validation samples (shorter set), therfore less batches. all batches evaluated per epoch.

                x_validation_batch         = x_validation_batch.to(device)
                y_validation_targets_batch = y_validation_targets_batch.to(device)

                y_validation_predictions_batch = model(x_validation_batch)

                # Calculate the validation loss
                validation_batch_loss = criterion(
                    y_validation_predictions_batch,
                    y_validation_targets_batch)

                epoch_validation_loss_sum += validation_batch_loss.item()

        epoch_validation_loss_avg = epoch_validation_loss_sum / len(validation_batches)
        validation_losses.append(epoch_validation_loss_avg)


        # Early stopping evaluation
        if epoch_validation_loss_avg < best_validation_loss:        # can add minimum improvement threshold later

            # Update best values
            best_validation_loss = epoch_validation_loss_avg
            best_epoch = epoch + 1  # epochs = 0,1,2,3... but reported as epochs = 1,2,3...

            # Store weights of the best state in a dictionary for later reuse, detach and clone to create independent copies
            best_state_dict = {parameter: value.detach().clone() for parameter, value in model.state_dict().items()}

            # Reset counter
            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience_epochs:
            break


        # Epochs in the model = 0,1,2 ... but usually reported as = 1,2,3
        print(
            f"Epoch: {epoch + 1:3d} | "
            f"Train loss: {epoch_training_loss_avg:.8f} | "
            f"Val loss: {epoch_validation_loss_avg:.8f}"
        )


    ############### Plot the loss curve
    plt.plot(training_losses, label = "Training Loss")
    plt.plot(validation_losses, label = "Validation Loss")
    plt.ylabel("Loss, Epoch avg.")
    plt.xlabel("Epoch")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    print(f'Early stopping after {best_epoch} training epochs.')

    return best_state_dict