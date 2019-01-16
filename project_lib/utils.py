import os
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from project_lib.dataset import StockDataSet
from project_lib.models import LSTM_Model
from project_lib.plotting import plot_cumsums

# Device agnostic PyTorch setup
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
else:
    device = torch.device('cpu')
    torch.backends.cudnn.enabled = False

CWD = os.getcwd()
DATA_CLEAN_DIR = os.path.join(CWD, 'data/Clean/')


def predict_stock(test_window, model, stock_name, window_size, num_values_to_predict):
    input_test = test_window[stock_name].iloc[:window_size-1].values
    input_test = torch.Tensor(input_test).reshape(1,-1,1)

    predictions = []
    for i in range(num_values_to_predict):
        hidden = model.init_hidden(1)
        pred = model(input_test, hidden)[0][0].detach().cpu().numpy()
        pred_sign = np.sign(pred)
        predictions.append(int(pred_sign))
        input_test = torch.cat([input_test, torch.Tensor([pred_sign]).reshape(1,1,1).float()], dim=1)[:,1:,:]
    predictions = np.array(predictions)

    pred_indices = test_window.index[window_size-1:]
    return pd.Series(predictions, index=pred_indices).rename(stock_name + '_pred')

def sliding_predictions(df, train_samples=500, window_size=25):
    batch_size = 32
    num_samples = len(df)
    print('Total samples: {}'.format(num_samples))

    all_preds = pd.DataFrame()
    all_cumsum = pd.DataFrame()

    # First sliding window start and end indices
    start_idx_train = 0
    end_idx_train   = start_idx_train + train_samples
    start_idx_test  = end_idx_train
    end_idx_test    = start_idx_test + (2*window_size)

    # Do the train / test loop for every sliding window
    while end_idx_test <= num_samples-1:
        print('Window incides -> Train: {}-{}, Test: {}-{}'.format(start_idx_train, end_idx_train,
                                                                   start_idx_test, end_idx_test))

        # Define the train window and create a data loader
        train_df = df.iloc[start_idx_train:end_idx_train]
        train_dataset = StockDataSet(train_df, window_size=window_size, step_size=1)
        train_loader  = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=8, drop_last=True, pin_memory=True)

        # Initialize the model
        model = LSTM_Model(1, hidden_size=64, num_layers=1, dropout=0)
        criterion = torch.nn.MSELoss()
        learning_rate = 5e-3
        weight_decay = 1e-3 # L2 regularizer parameter
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if torch.cuda.is_available():
            model.cuda()
            criterion.cuda()

        # Train the model

        num_epochs = 50
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                hidden = model.init_hidden(batch_size)
                inputs = inputs.reshape(batch_size, window_size-1, 1).float().to(device)
                labels = labels.to(device)
                # Forward pass
                preds = model(inputs, hidden)
                loss = criterion(preds, labels.reshape(-1,1).float())
                # Backward pass
                loss.backward()
                optimizer.step()

        # Predict values
        test_df = df.iloc[start_idx_test:end_idx_test-1]
        preds = []
        for stock_name in ['ABBN', 'CSGN', 'NESN', 'NOVN']:
            preds.append(predict_stock(test_df, model, stock_name, window_size, window_size))
        preds = pd.DataFrame(preds).transpose()
        all_preds = pd.concat([all_preds, preds])
        all_cumsum = pd.concat([all_cumsum, preds.cumsum()])

        # Shift sliding window forward
        start_idx_train = start_idx_train + window_size
        end_idx_train   = start_idx_train + train_samples
        start_idx_test  = end_idx_train
        end_idx_test    = start_idx_test + (2*window_size)

    return all_preds, all_cumsum

def plot_rolling_window(df_sign, year, train_samples=500, window_size=25):
    stocks = ['ABBN', 'CSGN', 'NESN', 'NOVN']
    stocks_pred = ['ABBN_pred', 'CSGN_pred', 'NESN_pred', 'NOVN_pred']

    start_date = pd.to_datetime('{}-01-01'.format(year))
    end_date = start_date + datetime.timedelta(days=250)
    df = df_sign.copy().loc[start_date:end_date]

    preds, cumsum_preds = sliding_predictions(df, train_samples=train_samples, window_size=window_size)

    preds.to_csv(os.path.join(DATA_CLEAN_DIR, 'preds_{}.csv.gz'.format(year)), compression='gzip')
    cumsum_preds.to_csv(os.path.join(DATA_CLEAN_DIR, 'cumsum_preds_{}.csv.gz'.format(year)), compression='gzip')

    cumsum_df = df.join(cumsum_preds).iloc[train_samples:]
    cumsum_df[stocks] = cumsum_df[stocks].cumsum()
    plot_cumsums(cumsum_df, year, window_size)

def evaluate(real_data, prediction):
    '''
    Verifies how much was won or lost during prediction period, by computing the pointwise
    product between the prediction and the difference of the real price, then summing up.

    Parameters
    ----------
    real_data:
        Dataframe containing the real prices.
    prediction:
        Dataframe containing the sign of the prediction.

    Returns
    -------
    result: float
        sum of pointwise multiplication
    '''
    # Index of dataframes to datetime
    real_data.index = pd.to_datetime(real_data.index)
    prediction.index = pd.to_datetime(prediction.index)
    prediction = prediction.iloc[24::25,:]
    prediction.index = prediction.index.date
    prediction = np.sign(prediction)
    #New dataframe with index of prediction
    verification = prediction.join(real_data,how='left')
    verification[real_data.columns] = verification[real_data.columns].diff()
    verification = verification.dropna()
    result = (verification[prediction.columns].values * verification[real_data.columns]).sum().sum()
    return result
