from torch import nn


class LSTM_ForecastModel(nn.Module):

    def __init__(self, num_input_features: int, lstm_width: int, lstm_depth: int, forecast_length: int):

        self.num_input_features = num_input_features   # multivariate input sequence: channel_number = features processed per timestep
        self.lstm_width         = lstm_width           # neurons per layer
        self.lstm_depth         = lstm_depth           # number of layers
        self.forecast_length    = forecast_length      # [hours] -- can be tuned as hyperparameter

        super().__init__()      # initializing the nn.Module parent class also


        # LSTM part of the NN
        self.lstm_network = nn.LSTM(
            input_size   = self.num_input_features,
            hidden_size  = self.lstm_width,
            num_layers   = self.lstm_depth,
            batch_first  = True,                         # batch size = first entry in the DataLoader outputs, typically True
            # dropout     = dropout       # 0.2 would mean roughly (20)% of the relevant activations are dropped during each training pass. can help avoid overfitting with large datasets
        )

        # Linear output layer for direct one-shot multi-timestep forecasting
        # --> Use only the output from the last timestep as projection to accumulate info in the LSTM state, discard other outputs --> see forward()
        self.output_layer = nn.Linear(
            in_features  = self.lstm_width,          # connecting to the last LSTM layer
            out_features = self.forecast_length      # projecting to the full forcast window at the same time
        )


    def forward(self, x_inputs):
        # perform the forward pass on itself by feeding inputs x
        # tensor from DataLoader: x_inputs.shape = (batch_size, context_length, num_input_features)

        lstm_output, _ = self.lstm_network(x_inputs)          # return structure: output, (h_n, c_n) = lstm(x) // output.shape == (batch_size, context_length, hidden_size)

        # select the last output for forecasting only, but all related batches and hidden features
        lstm_last_output = lstm_output[ : , -1, : ]    # indexation in the tensor: output[batch_index, time_index, hidden_size], starting with 0 -- means [:,context_length -1,:] or just [:,-1,:], remember counting starts with 0 so the last is context -1

        # make the forecasting projection with the last lstm outputs to the whole forecast horizon
        y_predictions = self.output_layer(lstm_last_output)

        return y_predictions     # shape: [batch_size, forecast_length]