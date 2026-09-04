# ***longdur_ett*** | Long-Duration Time Series Forecasting on the ETT Dataset
- (c) Johannes Aurich, August 2026 
- GitHub: https://github.com/jauric
- This project is licensed under the MIT License. See `LICENSE.md` for details.

## Overview

The initial version of this package was developed during the 2026 Summer School "Deep Learning for Time Series Modeling" at TU Berlin. It focuses on forecasting the ETT transformer oil temperature data set over long-durations with different modeling approaches.

The source dataset and background information can be found here:
https://github.com/zhouhaoyi/ETDataset

The first released version v1.0.0 of this package is an update of what has been developed during the Summer School courses. A summary slide deck of the initial, now obsolete results submitted for project grading can be found in the *results/summary_slides* folder, together with an updated version of that document.

Currently, only a single LSTM-based model and problem set has been implemented and compared to an ARIMA baseline. Future work and alternative approaches are listed in the roadmap below and the current summary slides.

Available materials focus on **one-shot univariate forecasting of the oil temperature with multivariate inputs**. 

## Contents

- ***models/*** : stores PyTorch model files
- ***utils/*** : scripts for data processing, loading, model training, and evaluation
- ***results/*** : used for all completed analysis (as .ipynb), related weights, and summaries
- ***notebooks/*** : contains template notebooks for different models and forecasting problems


## Installation & Requirements

### Minimum Requirements

- Python: 3.13.11+
- A suitable `torch` build is required (CPU or GPU)
- See `pyproject.toml` for all dependencies 

### Quick Start

Run these commands from a terminal inside the project root:

```bash
# Clone the repository
git clone https://github.com/jauric/longdur_ett.git
cd longdur_ett
# Create and activate a virtual environment and install the package
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Installation may fail due to insufficient temporary storage (large PyTorch wheels). Fix: 
# (1) Create a new temporary storage folder on your home directory
# (2) Select this folder for temporary storage needs of the current terminal
# (3) Retry installation
mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"
pip install -e .

# Open the example notebook with Jupyter
jupyter lab notebooks/ETTh1_forecasting_template.ipynb

# Or inspect the results notebook
jupyter lab results/ETTh1_96-16_LSTM_1x48.ipynb
```


## User Guidance
- Jupyter notebook templates in the *notebooks/* folder can be copied and adapted to the desired problem
- After results have been obtained, export the weights and store notebook and weights in the results folder
- An example is available as "ETTh1_96-16_LSTM_1x48.ipynb"
- The model will automatically train/evaluate on cuda if suitable GPUs are available

## Future Improvement Roadmap

- Run ARIMA on every test sample (not just the first) for full comparability of MSE, MAE
- Remove scikit-learn dependency, implement mse, mae, scalers in numpy instead
- Implement k-fold cross-validation for hyperparameter optimization
- Add state space models (S4, Mamba) 
- Add foundation models (xLSTM)
- Add DLinear and an adaptation to LSTMs ("DLSTM")
- Add forecast-correction strategies with recursive loops for dynamic updates
- Reverse-engineer physical causalities within the system to reduce noise (ambient conditions, heat transfer, ...)
- Add minimum improvement threshold to early stopping evaluations in training.training_loop()
- Improve remote GPU support for faster training & extensive hyperparameter search
- Improve reproducibility through saved seeds
