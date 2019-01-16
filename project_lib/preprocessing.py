import os
import pandas as pd
import numpy as np
import datetime
pd.options.mode.chained_assignment = None  # default='warn'


def preprocess(stocks, directory, verbosity=0):
    '''
    Function takes names of stock and root data directory as argument and returns
    preprocessed dataframe containing one column per stock, minute by minute data
    of business hours keeping only the sign, resampled to full hours.
    Various parts of the function are hard coded, which is due to the fact that the
    data was not collected by ourselves and thus has some quirks to it.

    Parameters
    ----------
    stocks: array_like
        List containing stock name strings
    directory: string
        Path to stock files.
    verbosity: int, optional
        Level of print output. 0 for no prints, 1 for few, 2 for all.

    Returns
    -------
    cleaned_df: Pandas DataFrame
        The preprocessed and cleaned dataframe.
    '''
    df_stock = pd.DataFrame()
    for stock in stocks:
        filenames = sorted(os.listdir(os.path.join(directory, stock)))
        for filename in filenames:
            if verbosity >= 2:
                print(os.path.join(os.path.join(directory, stock), filename))
            df = pd.read_csv(os.path.join(os.path.join(directory, stock), filename), compression='gzip')
            df = df.rename(columns={' bid':'bid','#date_xl':'date',' bidQ':'bidQ',' ask':'ask',' askQ':'askQ',' ts':'second'})

            df = datetime_index_formatting(df)
            df = bid_ask_formatting(df)
            df = resampling(df)
            df = log_returns(df)

            df_stock = df_stock.append(df)

        if stock == stocks[0]:
            df_stock = df_stock.rename(columns = {'bid-ask':stock})
            df_total = df_stock
            df_stock = pd.DataFrame()
        else:
            df_total[stock] = df_stock['bid-ask']
            df_stock = pd.DataFrame()

        if verbosity >= 1:
            print('Processed stock {}'.format(stock))

    return df_total.dropna()

def excel_to_date(ordinal):
    '''
    Function takes an ordinal and returns the correct timestamp from excel data.

    Parameters
    ----------
    ordinal: int
        Ordinal datetime

    Returns
    -------
    datetime: Timestamp
        The correct timestamp object.
    '''
    excel = datetime.datetime(1900,1,1,0,0).toordinal()
    excel_bug = 2
    return pd.Timestamp.fromordinal(excel-excel_bug+int(ordinal))

def datetime_index_formatting(df):
    '''
    Function takes date and time, creates a datetime object and sets it as the index of the df.

    Parameters
    ----------
    df: Pandas DataFrame
        Dataframe to index by datetime.

    Returns
    -------
    df: Pandas DataFrame
        Correcly indexed Dataframe.
    '''
    df['datetime'] = df['date'].apply(excel_to_date) + pd.to_timedelta(df['second'], unit='s')
    df['datetime'].dt.tz_localize('Europe/Zurich')
    df = df.set_index(pd.to_datetime(df['datetime']))
    df = df.drop(['date', 'second','datetime'], axis=1)
    return df

def bid_ask_formatting(df):
    '''
    Function computes average best offer of ask and bid.

    Parameters
    ----------
    df: Pandas DataFrame
        Dataframe containing ask and bid prices.

    Returns
    -------
    df: Pandas DataFrame
        Dataframe containing only average best offer.
    '''
    df = df[(df['lastQ'] != -1)]
    df['bid-ask'] = (df['ask'] + df['bid']) / 2
    df = df.drop(['ask','askQ','bid','bidQ','last','lastQ'],axis=1)
    return df

def resampling(df):
    '''
    Function resamples by hour, then keeps only business hours and drops nans.

    Parameters
    ----------
    df: Pandas DataFrame
        Dataframe to be resampled hourly.

    Returns
    -------
    df: Pandas DataFrame
        Resampled Dataframe.
    '''
    df = df.resample("1H").mean()
    df = df.between_time('09:30', '15:59')
    df = df.dropna()
    return df

def log_returns(df):
    '''
    Function computes log returns

    Parameters
    ----------
    df: Pandas DataFrame
        Dataframe with prices.

    Returns
    -------
    df: Pandas DataFrame
        Log returns of prices.
    '''
    df['bid-ask'] = np.log(df['bid-ask']).diff()
    df = df.iloc[1:]
    return df
