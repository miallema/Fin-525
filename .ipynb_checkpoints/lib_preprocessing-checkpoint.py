import os
import pandas as pd
import numpy as np
import datetime
pd.options.mode.chained_assignment = None  # default='warn'


#Function takes names of stock and root data directory as argument and returns 
#preprocessed dataframe containing one column per stock, minute by minute data
#of business hours keeping only the sign.
#Various parts of the function are hard coded, which is due to the fact that the
#data was not collected by ourselves and thus has some quirks to it.
def preprocess(stocks, directory):
    df_stock = pd.DataFrame()
    for stock in stocks:
        filenames = sorted(os.listdir(os.path.join(directory, stock)))  
        for filename in filenames: 
            print(os.path.join(os.path.join(directory, stock), filename))
            df = pd.read_csv(os.path.join(os.path.join(directory, stock), filename),compression='gzip')
            df = df.rename(columns={' bid':'bid','#date_xl':'date',' bidQ':'bidQ',' ask':'ask',' askQ':'askQ',' ts':'second'})

            df = datetime_index_formatting(df)
            df = bid_ask_formatting(df)
            df = resampling(df)
            df = sign_log_returns(df)
        
            df_stock = df_stock.append(df)

        if stock == stocks[0]:
            df_stock = df_stock.rename(columns = {'bid-ask':stock})
            df_total = df_stock
            df_stock = pd.DataFrame()
        else:
            df_total[stock] = df_stock['bid-ask']
            df_stock = pd.DataFrame()
        
    return df_total.dropna()            
  
    
#Function takes an ordinal and returns the correct timestamp from excel data
def excel_to_date(ordinal):
    excel = datetime.datetime(1900,1,1,0,0).toordinal()
    excel_bug = 2
    return pd.Timestamp.fromordinal(excel-excel_bug+int(ordinal))           
            

#Function takes date and time, creates a datetime object and sets it as the index of the df.
def datetime_index_formatting(df):
    df['datetime'] = df['date'].apply(excel_to_date) + pd.to_timedelta(df['second'], unit='s')
    df['datetime'].dt.tz_localize('Europe/Zurich')
    df = df.set_index(pd.to_datetime(df['datetime']))
    df = df.drop(['date', 'second','datetime'], axis=1) 
    return df


#Function computes weighted average of ask and bid
def bid_ask_formatting(df):
    df = df[(df['lastQ'] != -1)]
    df['bid-ask'] = (df['ask']*df['askQ'] + df['bid']*df['bidQ']) / (df['askQ'] + df['bidQ'])
    df = df.drop(['ask','askQ','bid','bidQ','last','lastQ'],axis=1)  
    return df


#Function resamples by minute, then keeps only business hours and drops nans.
def resampling(df):
    df = df.resample("1Min").mean()
    df = df.between_time('09:30', '15:59')
    df = df.dropna()    
    return df
   
    
#Function normalizes, computes log returns and keeps only the sign.
def sign_log_returns(df):
    df['bid-ask'] = df['bid-ask'] / df['bid-ask'][0] * 100
    df['bid-ask'] = np.log(df['bid-ask']).diff()
    df['bid-ask'] = np.sign(df['bid-ask'])
    df = df.iloc[1:]  
    return df