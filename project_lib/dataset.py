import torch
from torch.utils.data import Dataset


class StockDataSet(Dataset):
    '''
    Dataset containing training data. Takes a dataframe with multiple stock
    log-returns and creates training data input and target samples by sliding
    a window with given size over all stocks, shifting it at each step by
    a given step size. The last element is always taken to be the target.

    Parameters
    ----------
    df: Pandas DataFrame
        Dataframe to be used for dataset.
    window_size: int, optional
        x
    step_size: int, optional
        x
    '''
    def __init__(self, df, window_size=15, step_size=5):
        self.datetime = df.index
        self.data = []

        for stock_name in df.columns:
            for window_start in range(0, len(df) - window_size, step_size):
                w = df[stock_name].iloc[window_start: window_start + window_size]
                self.data.append(w.values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        window = self.data[idx]
        series = window[:-1]
        target = window[-1]

        return series, target
