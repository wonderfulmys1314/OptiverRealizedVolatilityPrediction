# -*- coding:utf-8 -*-

"""
    copy:https://www.kaggle.com/tensorchoko/optiver-realized-lightgbm-for-beginner
"""

from pathlib import Path 
import pandas as pd 
import glob 
import numpy as np
import fastparquet as fpa 
import seaborn as sns 
import matplotlib.pyplot as plt 
import tqdm
from tqdm.notebook import tqdm
import gc 
from sklearn.preprocessing import MinMaxScaler 
import warnings 
warnings.filterwarnings('ignore')
import os 


def calc_wap(df):
    return (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1'])/(df['bid_size1'] + df['ask_size1'])

def calc_wap2(df):
    return (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2'])/(df['bid_size2'] + df['ask_size2'])

def calc_wap3(df):
    return (df['bid_price2'] * df['bid_size2'] + df['ask_price2'] * df['ask_size2']) / (df['bid_size2']+ df['ask_size2'])

def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff() 

def realized_volatility(series):
    return np.sqrt(np.sum(series**2))

def count_unique(series):
    return len(np.unique(series))


def preprocessor_trade(file_path):
    df = pd.read_parquet(file_path)
    df['log_return'] = df.groupby('time_id')['price'].apply(log_return)

    aggregate_dictionary = {
        'log_return':[realized_volatility],
        'seconds_in_bucket':[count_unique],
        'size':[np.sum],
        'order_count':[np.sum]
    }


    df_feature = df.groupby('time_id').agg(aggregate_dictionary)

    df_feature = df_feature.reset_index()
    df_feature.columns = ['_'.join(col) for col in df_feature.columns]

    last_seconds = [300]

    for second in last_seconds:
        second = 600 - second
        df_feature_sec = df.query(f'seconds_in_bucket >= {second}').groupby('time_id').agg(aggregate_dictionary)
        df_feature_sec = df_feature_sec.reset_index()
        df_feature_sec.columns = ['_'.join(col) for col in df_feature_sec.columns]
        df_feature_sec = df_feature_sec.add_suffix('_' + str(second))
        df_feature = pd.merge(df_feature,df_feature_sec,how='left',left_on='time_id_',right_on=f'time_id__{second}')

        df_feature = df_feature.drop([f'time_id__{second}'],axis=1)

    df_feature = df_feature.add_prefix("trade_")
    stock_id = file_path.split('/')[-2].split("=")[1]
    df_feature['row_id'] = df_feature['trade_time_id_'].apply(lambda x: f'{stock_id}-{x}')
    df_feature = df_feature.drop(['trade_time_id_'],axis=1)
    
    print(df_feature)

    return df_feature


def preprocessor_book(file_path):
    df = pd.read_parquet(file_path)

    #calculate return etc
    df['wap'] = calc_wap(df)
    df['log_return'] = df.groupby('time_id')['wap'].apply(log_return)
    
    df['wap2'] = calc_wap2(df)
    df['log_return2'] = df.groupby('time_id')['wap2'].apply(log_return)
    
    df['wap3'] = calc_wap3(df)
    df['log_return3'] = df.groupby('time_id')['wap3'].apply(log_return)
    
    df['wap_balance'] = abs(df['wap'] - df['wap2'])
    
    df['price_spread'] = (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1'])/2)
    df['bid_spread'] = df['bid_price1'] - df['bid_price2']
    df['ask_spread'] = df['ask_price1'] - df['ask_price2']
    df['total_volume'] = (df['ask_size1'] + df['ask_size2']) + (df['bid_size1'] + df['bid_size2'])
    df['volume_imbalance'] = abs((df['ask_size1'] + df['ask_size2']) - (df['bid_size1'] + df['bid_size2']))

    #dict for aggregate
    create_feature_dict = {
        'log_return':[realized_volatility],
        'log_return2':[realized_volatility],
        'log_return3':[realized_volatility],
        'wap_balance':[np.mean],
        'price_spread':[np.mean],
        'bid_spread':[np.mean],
        'ask_spread':[np.mean],
        'volume_imbalance':[np.mean],
        'total_volume':[np.mean],
        'wap':[np.mean],
    }
    
    def get_stats_window(seconds_in_bucket, add_suffix = False):
        # Group by the window
        df_feature = df[df['seconds_in_bucket'] >= seconds_in_bucket].groupby(['time_id']).agg(create_feature_dict).reset_index()
        # Rename columns joining suffix
        df_feature.columns = ['_'.join(col) for col in df_feature.columns]
        # Add a suffix to differentiate windows
        if add_suffix:
            df_feature = df_feature.add_suffix('_' + str(seconds_in_bucket))
        return df_feature    
    
    # Get the stats for different windows
    df_feature = get_stats_window(seconds_in_bucket = 0, add_suffix = False)
    df_feature_450 = get_stats_window(seconds_in_bucket = 450, add_suffix = True)
#     df_feature_500 = get_stats_window(seconds_in_bucket = 500, add_suffix = True)
#     df_feature_400 = get_stats_window(seconds_in_bucket = 400, add_suffix = True)
    df_feature_300 = get_stats_window(seconds_in_bucket = 300, add_suffix = True)
#     df_feature_200 = get_stats_window(seconds_in_bucket = 200, add_suffix = True)
    df_feature_150 = get_stats_window(seconds_in_bucket = 150, add_suffix = True)

    # Merge all
    df_feature = df_feature.merge(df_feature_450, how = 'left', left_on = 'time_id_', right_on = 'time_id__450')
    df_feature = df_feature.merge(df_feature_300, how = 'left', left_on = 'time_id_', right_on = 'time_id__300')
#     df_feature = df_feature.merge(df_feature_300, how = 'left', left_on = 'time_id_', right_on = 'time_id__300')
    df_feature = df_feature.merge(df_feature_150, how = 'left', left_on = 'time_id_', right_on = 'time_id__150')
#     df_feature = df_feature.merge(df_feature_100, how = 'left', left_on = 'time_id_', right_on = 'time_id__100')
    # Drop unnecesary time_ids
    df_feature.drop(['time_id__450', 'time_id__300', 'time_id__150'], axis = 1, inplace = True)
    
    
    # Create row_id so we can merge
    stock_id = file_path.split('/')[-2].split("=")[1]
    print(df_feature.head(10))
    df_feature['row_id'] = df_feature['time_id_'].apply(lambda x: f'{stock_id}-{x}')
    df_feature = df_feature.drop(['time_id_'],axis=1)

    # print(df_feature)
    
    return df_feature

def preprocessor(list_stock_ids, is_train = True):
    from joblib import Parallel, delayed # parallel computing to save time
    df = pd.DataFrame()
    
    def for_joblib(stock_id):
        if is_train:
            file_path_v1 = data_dir + "book_train.parquet/stock_id=" + str(stock_id)
            file_path_book = file_path_v1 + "/" + os.listdir(file_path_v1)[0]
            file_path_v2 = data_dir + "trade_train.parquet/stock_id=" + str(stock_id)
            file_path_trade = file_path_v2 + "/" + os.listdir(file_path_v2)[0]
        else:
            # file_path_book = data_dir + "book_test.parquet/stock_id=" + str(stock_id)
            # file_path_trade = data_dir + "trade_test.parquet/stock_id=" + str(stock_id)
            file_path_v1 = data_dir + "book_test.parquet/stock_id=" + str(stock_id)
            file_path_book = file_path_v1 + "/" + os.listdir(file_path_v1)[0]
            file_path_v2 = data_dir + "trade_test.parquet/stock_id=" + str(stock_id)
            file_path_trade = file_path_v2 + "/" + os.listdir(file_path_v2)[0]
            
        df_tmp = pd.merge(preprocessor_book(file_path_book),preprocessor_trade(file_path_trade),on='row_id',how='left')
     
        return pd.concat([df,df_tmp])
    
    df = Parallel(n_jobs=-1, verbose=1)(
        delayed(for_joblib)(stock_id) for stock_id in list_stock_ids
        )

    df =  pd.concat(df,ignore_index = True)
    return df

data_dir = "../input/optiver-realized-volatility-prediction/"

train = pd.read_csv(data_dir + 'train.csv')
train_ids = train.stock_id.unique()
df_train = preprocessor(list_stock_ids=train_ids, is_train = True)
train['row_id'] = train['stock_id'].astype(str) + '-' + train['time_id'].astype(str)
train = train[['row_id','target']]
df_train = train.merge(df_train, on = ['row_id'], how = 'left')

test = pd.read_csv(data_dir + 'test.csv')
df_test_ids = test.stock_id.unique()
df_test = preprocessor(list_stock_ids=df_test_ids, is_train = False)
df_test = test.merge(df_test, on = ['row_id'], how = 'left')


from sklearn.model_selection import KFold


df_train['stock_id'] = df_train['row_id'].apply(lambda x:x.split('-')[0])
df_test['stock_id'] = df_test['row_id'].apply(lambda x:x.split('-')[0])
stock_id_target_mean = df_train.groupby('stock_id')['target'].mean() 
df_test['stock_id_target_ext'] = df_test['stock_id'].map(stock_id_target_mean) # test_set

tmp = np.repeat(np.nan, df_train.shape[0])
kf = KFold(n_splits = 10, shuffle=True,random_state = 77)
# 分10次，每一个时间段的值使用其他时段的平均值
for idx_1, idx_2 in tqdm(kf.split(df_train)):
    target_mean = df_train.iloc[idx_1].groupby('stock_id')['target'].mean()

    tmp[idx_2] = df_train['stock_id'].iloc[idx_2].map(target_mean)
df_train['stock_id_target_ext'] = tmp

# 特征和标签
df_train = df_train.reset_index(drop=True)
target = df_train['target']
df_train = df_train.drop(['target'],axis=1)
X = df_train
y = target

col =['log_return_realized_volatility',
    'log_return2_realized_volatility', 'log_return3_realized_volatility',
    'wap_balance_mean', 'price_spread_mean', 'bid_spread_mean',
    'ask_spread_mean', 'volume_imbalance_mean', 'total_volume_mean',
    'wap_mean', 'log_return_realized_volatility_450',
    'log_return2_realized_volatility_450',
    'log_return3_realized_volatility_450', 'wap_balance_mean_450',
    'price_spread_mean_450', 'bid_spread_mean_450', 'ask_spread_mean_450',
    'volume_imbalance_mean_450', 'total_volume_mean_450', 'wap_mean_450',
    'log_return_realized_volatility_300',
    'log_return2_realized_volatility_300',
    'log_return3_realized_volatility_300', 'wap_balance_mean_300',
    'price_spread_mean_300', 'bid_spread_mean_300', 'ask_spread_mean_300',
    'volume_imbalance_mean_300', 'total_volume_mean_300', 'wap_mean_300',
    'log_return_realized_volatility_150',
    'log_return2_realized_volatility_150',
    'log_return3_realized_volatility_150', 'wap_balance_mean_150',
    'price_spread_mean_150', 'bid_spread_mean_150', 'ask_spread_mean_150',
    'volume_imbalance_mean_150', 'total_volume_mean_150', 'wap_mean_150',
    'trade_log_return_realized_volatility',
    'trade_seconds_in_bucket_count_unique', 'trade_size_sum',
    'trade_order_count_mean', 'trade_log_return_realized_volatility_300',
    'trade_seconds_in_bucket_count_unique_300', 'trade_size_sum_300',
    'trade_order_count_mean_300',  'stock_id_target_ext'
]

from sklearn.model_selection import train_test_split, KFold


from sklearn.model_selection import train_test_split
import lightgbm as lgb
best_lgb_params = {
 'metric': 'mae',
 'objective': 'regression'}
best_lgb_params["learning_rate"] = 0.005
best_lgb_params["early_stopping_round"] = 50  
best_lgb_params["max_depth"] = 100
best_lgb_params["num_iterations"] = 2500


import optuna 
import optuna.integration.lightgbm as lgbo

params = { 'objective': 'mean_squared_error', 'metric': 'rmse' }
from sklearn.metrics import mean_absolute_error

x_train, x_test, y_train, y_test = train_test_split(X[col], y, test_size=0.3, random_state=42)
lgb_train = lgb.Dataset(x_train, y_train)
lgb_valid = lgb.Dataset(x_test, y_test)

model = lgbo.train(params, lgb_train, valid_sets=[lgb_valid], verbose_eval=False, num_boost_round=100, early_stopping_rounds=5) 
model.params



