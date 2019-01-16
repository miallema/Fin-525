# Financial Big Data

This Repository contains the code from the Course FIN-525.

## Overview

Our code is organized as such: 

- ```project.ipynb```: The main Jupyter Notebook containing the analysis
- ```project_lib/```: Library containing code used multiple times in our notebook
  - ```dataset.py```: Contains PyTorch Dataset definition for training the model
  - ```models.py```: Contains the LSTM model definition
  - ```plotting.py```: Contains various plotting functions
  - ```preprocessing.py```: Contains functions used for data cleaning and preprocessing
  - ```utils.py```: Contains various functions to predict using a sliding window
- ```data/```: Directory where data is stored
  - ```Raw/```: Place the provided 12GB tick-by-tick assets in here
  - ```Clean/```: Folder containing cleaned data
- ```plots/```: Directory where plots will be saved as PDFs

## Requirements

The project was tested with the following libraries:

- torch==1.0.0
- numpy==1.15.4
- pandas==0.23.4
- pandas-datareader==0.7.0
- statsmodels==0.9.0
- matplotlib==3.0.2
