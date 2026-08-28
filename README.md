# ETTh1 - Long-Term Time Series Modelling with Neural Nets
(c) Johannes Aurich, August 2026

This material was developed during the 2026 Summer School "Deep Learning for Time Series Modelin" at TU Berlin. It is concerned with modeling the ETTh1 transformer oil temperature problem for long-duration forecasting.

The source dataset and background information can be found here:
https://github.com/zhouhaoyi/ETDataset

## Contents for this folder

- README.md
- 260228_ETTh1_Final Presentation.pdf
- main (contains model weights and Jupyter notebooks for analysis and model training)
    - LSTM_1x7_weights.pt
    - LSTM_5x7_weights.pt
    - LSTM_2x17_weights.pt
    - ARIMA_model.ipynb
    - LSTM_1x7_model.ipynb
    - LSTM_5x7_model.ipynb
    - LSTM_2x17_model.ipynb
    - data_analysis.ipynb

## Installation

For installation in a local virtual environment, run from your local folder:

python -m pip install -r requirements.txt

Recommended use is through VS Code and its Jupyter Notebooks addon. Check the Jupyter Notebook page in VS Code's extension collection for more information about installation.

## Using the Jupyter Notebooks

Each notebook is self-sustained and contains all information required to train and evaluate the respective model configuration. 

A notebook has two modes: "training" and "testing", to be set at the top. It contains multtwo hard-coded runtime breaks to further streamline the workflow.

- Training: 
    - initiates model training based on the specified main hyperparameters defined at the notebook top. Testing sections will not be access when the whole notebook is in "run all" mode. 
    After training, there is a hard-coded RuntimeError to stop the whole file from-auto executing (when in "run all" mode). This allows to analyze the training loss plot and choose the ideal number of training epochs before engaging in final retraining. 
    - Training weights can be exported.

- Testing
    - Another RuntimeError breakes "run all" before entering the testing sections. Change mode to "testing" at the top or run each window manually.

## Improvement Roadmap

- Restructure code into importable .py modules and integrating .ipynb notebooks
- Trained model weights are not explicitly purged after the training/validation loops. When engaging retraining for the optimal number of epochs, this may currently just lead to further training on the same parameters and thereby overfitting. This may be a key reason for the bad model performance. To be analyzed.


