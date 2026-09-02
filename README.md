# ETTh1 - Long-Term Time Series Modelling with Neural Nets
(c) Johannes Aurich, August 2026

This material was developed during the 2026 Summer School "Deep Learning for Time Series Modelin" at TU Berlin. It is concerned with modeling the ETTh1 transformer oil temperature problem for long-duration forecasting.

The source dataset and background information can be found here:
https://github.com/zhouhaoyi/ETDataset

## Contents for this folder

- DL4TS/
    - README.md
    - pyproject.toml
    - .gitignore
    - models/
        - lstm_model.py
    - weights/
        - (...).pt
    - utils/
        - data_loader.py
        - data_utils.py
        - training.py
        - evaluation.py
    - notebooks/
        - (...).ipynb
    - docs/
        - (final_presentation).pdf



## Future improvement

- add minimum improvement threshold to early stopping evaluations in training.training_loop()
- improve reproducibility through saved seeds
- use k-fold cross-validation for hyperparameter optimization