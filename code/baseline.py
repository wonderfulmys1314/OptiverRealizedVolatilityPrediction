# -*- coding -*-

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import glob

def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff() 

def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return**2))

def realized_volatility_per_time_id(file_path, prediction_column_name):
    df_book_data = pd.read_parquet(file_path)
    df_book_data['wap'] = ( df_book_data['bid_price1'] * df_book_data['ask_size1'] + df_book_data['ask_price1'] * df_book_data['bid_size1'])  / ( df_book_data['bid_size1'] + df_book_data['ask_size1'])
    df_book_data['log_return'] = df_book_data.groupby(['time_id'])['wap'].apply(log_return)
    df_book_data = df_book_data[~df_book_data['log_return'].isnull()]
    df_realized_vol_per_stock =  pd.DataFrame(df_book_data.groupby(['time_id'])['log_return'].agg(realized_volatility)).reset_index()
    df_realized_vol_per_stock = df_realized_vol_per_stock.rename(columns = {'log_return':prediction_column_name})
    stock_id = file_path.split('=')[1]
    df_realized_vol_per_stock['row_id'] = df_realized_vol_per_stock['time_id'].apply(lambda x:f'{stock_id}-{x}')
    return df_realized_vol_per_stock[['row_id',prediction_column_name]]


def past_realized_volatility_per_stock(list_file,prediction_column_name):
    df_past_realized = pd.DataFrame()
    for file in list_file:
        df_past_realized = pd.concat(
            [df_past_realized,realized_volatility_per_time_id(file,prediction_column_name)])
    return df_past_realized


list_order_book_file_test = glob.glob('input/optiver-realized-volatility-prediction/book_test.parquet/*')

df_naive_pred_test = past_realized_volatility_per_stock(
    list_file=list_order_book_file_test,
    prediction_column_name='target')
df_naive_pred_test.to_csv('submission.csv',index = False)
