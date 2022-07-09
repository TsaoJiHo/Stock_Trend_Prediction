import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import json
import finlab
from finlab import data

import requests
import os
import sys
import subprocess

with open('config.json') as config_file:
    config = json.load(config_file)



if not config['LOCAL_MODE']:
    # check if the library folder already exists, to avoid building everytime you load the pahe
    if not os.path.isdir("/tmp/ta-lib"):

        # Download ta-lib to disk
        with open("/tmp/ta-lib-0.4.0-src.tar.gz", "wb") as file:
            response = requests.get(
                "http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz"
            )
            file.write(response.content)
        # get our current dir, to configure it back again. Just house keeping
        default_cwd = os.getcwd()
        os.chdir("/tmp")
        # untar
        os.system("tar -zxvf ta-lib-0.4.0-src.tar.gz")
        os.chdir("/tmp/ta-lib")
        os.system("ls -la /app/equity/")
        # build
        os.system("./configure --prefix=/home/appuser")
        os.system("make")
        # install
        os.system("make install")
        # back to the cwd
        os.chdir(default_cwd)
        sys.stdout.flush()

    # add the library to our current environment
    from ctypes import *

    lib = CDLL("/home/appuser/lib/libta_lib.so.0.0.0")
    # import library
    try:
        import talib
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--global-option=build_ext", "--global-option=-L/home/appuser/lib/", "--global-option=-I/home/appuser/include/", "ta-lib"])
    finally:
        import talib

SAMPLE_COUNT = 320

# https://ai.finlab.tw/api_token/
finlab.login('GSCV5+L51GA2wGPvFZYpqKai2rKE9Tw2eG1FVGPX+hzn2SF1oFvvSj1N7fCJLTrB#free')

class StockDataset(Dataset):
    ''' Dataset for loading and preprocessing the dataset '''
    def __init__(self,
                 df
                ):

        data = df.values.tolist()

        self.data = torch.FloatTensor(data)


        # Normalize features
        self.data = \
            (self.data - self.data.mean(dim=0, keepdim=True)) \
            / self.data.std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]

        # print(f'Finished reading the {mode} set of Dataset ({len(self.data)} samples found, each dim = {self.dim})')

    def __getitem__(self, index):

        return self.data[index]

    def __len__(self):

        return len(self.data)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.layer2 = nn.Linear(22, 256)
        self.layer2_bn=nn.BatchNorm1d(256)

        self.layer3 = nn.Linear(256, 128)
        self.layer3_bn=nn.BatchNorm1d(128)

        self.layer4 = nn.Linear(128, 64)
        self.layer4_bn=nn.BatchNorm1d(64)

        self.layer5 = nn.Linear(64, 32)
        self.layer5_bn=nn.BatchNorm1d(32)

        self.drop = nn.Dropout(0.5)

        self.out = nn.Linear(32, 3) 
        
        self.act_fn = nn.ReLU()
        

    def forward(self, x):

        x = self.layer2(x)
        x = self.layer2_bn(x)
        x = self.act_fn(x)
        x = self.drop(x)

        x = self.layer3(x)
        x = self.layer3_bn(x)
        x = self.act_fn(x)
        x = self.drop(x)
 
        x = self.layer4(x)
        x = self.layer4_bn(x)
        x = self.act_fn(x)
        x = self.drop(x)

        x = self.layer5(x)
        x = self.layer5_bn(x)
        x = self.act_fn(x)
        x = self.drop(x)

        x = self.out(x)

        
        return x

@st.cache
def features_df(newest_date):

    with open('codes_300.txt', 'r') as f:
        codes = f.read().split()

    rsi_14_df = data.indicator('RSI', timeperiod=14)[codes].tail(SAMPLE_COUNT)
    rsi_42_df = data.indicator('RSI', timeperiod=42)[codes].tail(SAMPLE_COUNT)

    k_9, d_9 = data.indicator('STOCH', timeperiod=9)
    k_9_df = k_9[codes].tail(SAMPLE_COUNT)
    d_9_df = d_9[codes].tail(SAMPLE_COUNT)

    k_27, d_27 = data.indicator('STOCH', timeperiod=27)
    k_27_df = k_27[codes].tail(SAMPLE_COUNT)
    d_27_df = d_27[codes].tail(SAMPLE_COUNT)

    macd = data.indicator('MACD')
    dif_df = macd[0][codes].tail(SAMPLE_COUNT)
    macd_df = macd[1][codes].tail(SAMPLE_COUNT)

    willr_14_df = data.indicator('WILLR', timeperiod=14)[codes].tail(SAMPLE_COUNT)
    willr_42_df = data.indicator('WILLR', timeperiod=42)[codes].tail(SAMPLE_COUNT)

    sma_20_df = data.indicator('SMA', timeperiod=20)[codes].tail(SAMPLE_COUNT)
    sma_60_df = data.indicator('SMA', timeperiod=60)[codes].tail(SAMPLE_COUNT)

    ema_20_df = data.indicator('EMA', timeperiod=20)[codes].tail(SAMPLE_COUNT)
    ema_60_df = data.indicator('EMA', timeperiod=60)[codes].tail(SAMPLE_COUNT)

    atr_14_df = data.indicator('ATR', timeperiod=14)[codes].tail(SAMPLE_COUNT)
    atr_42_df = data.indicator('ATR', timeperiod=42)[codes].tail(SAMPLE_COUNT)

    adx_14_df = data.indicator('ADX', timeperiod=14)[codes].tail(SAMPLE_COUNT)
    adx_42_df = data.indicator('ADX', timeperiod=42)[codes].tail(SAMPLE_COUNT)

    cci_14_df = data.indicator('CCI', timeperiod=14)[codes].tail(SAMPLE_COUNT)
    cci_42_df = data.indicator('CCI', timeperiod=42)[codes].tail(SAMPLE_COUNT)

    roc_10_df = data.indicator('ROC', timeperiod=10)[codes].tail(SAMPLE_COUNT)
    roc_30_df = data.indicator('ROC', timeperiod=30)[codes].tail(SAMPLE_COUNT)

    ret = []

    for i, code in enumerate(codes):

        columns = []
        columns.append(roc_10_df[code].rename('roc_10'))
        columns.append(roc_30_df[code].rename('roc_30'))
        columns.append(cci_14_df[code].rename('cci_14'))
        columns.append(cci_42_df[code].rename('cci_42'))
        columns.append(adx_14_df[code].rename('adx_14'))
        columns.append(adx_42_df[code].rename('adx_42'))
        columns.append(atr_14_df[code].rename('atr_14'))
        columns.append(atr_42_df[code].rename('atr_42'))
        columns.append(ema_20_df[code].rename('ema_20'))
        columns.append(ema_60_df[code].rename('ema_60'))
        columns.append(sma_20_df[code].rename('sma_20'))
        columns.append(sma_60_df[code].rename('sma_60'))
        columns.append(willr_14_df[code].rename('willr_14'))
        columns.append(willr_42_df[code].rename('willr_42'))
        columns.append(macd_df[code].rename('macd'))
        columns.append(dif_df[code].rename('dif'))
        columns.append(k_9_df[code].rename('k_9'))
        columns.append(d_9_df[code].rename('d_9'))
        columns.append(k_27_df[code].rename('k_27'))
        columns.append(d_27_df[code].rename('d_27'))
        columns.append(rsi_14_df[code].rename('rsi_14'))
        columns.append(rsi_42_df[code].rename('rsi_42'))
      
        ret.append((i + 1, pd.concat(columns, axis=1)))

    return ret

def main():


    date_series = data.indicator('ROC').tail(SAMPLE_COUNT).index.date

    current_date = st.selectbox(
     'Select Date',
     date_series,
     index = SAMPLE_COUNT - 1)
    
    st.subheader(f'Current Date Selection: {current_date}')
    st.write('\n')
    st.write('● Predicted Fluctuation Rate is considered as the stop-loss/take-profit rate.')
    st.write('● Softmax Value indicates the output of the deep model which takes technical indicators as input.')

    device = 'cpu'

    with open('codes_300.txt', 'r') as f:
        codes = f.read().split()

    with open('stocks_300.txt', 'r', encoding="big5", errors='ignore') as f:
        stocks = f.read().split()

    with open('trend_params_300.json') as jsonfile:
        code_para_map = json.load(jsonfile)

    predicted_label = {10: [[] for _ in range(SAMPLE_COUNT)], 20: [[] for _ in range(SAMPLE_COUNT)],
                    30: [[] for _ in range(SAMPLE_COUNT)], 40: [[] for _ in range(SAMPLE_COUNT)],
                    60: [[] for _ in range(SAMPLE_COUNT)], 120:[[] for _ in range(SAMPLE_COUNT)]}
                    
    predicted_value = {10: [[] for _ in range(SAMPLE_COUNT)], 20: [[] for _ in range(SAMPLE_COUNT)],
                    30: [[] for _ in range(SAMPLE_COUNT)], 40: [[] for _ in range(SAMPLE_COUNT)],
                    60: [[] for _ in range(SAMPLE_COUNT)], 120:[[] for _ in range(SAMPLE_COUNT)]}


    for i, df in features_df(date_series[-1]):
    # print(f'model {i}')
        for d in [10, 20, 30, 40, 60, 120]:
            
            model_path = f'./models_300/models/model_{i}_{d}d.ckpt'
            # create testing dataset

            test_set = StockDataset(df)
            test_loader = DataLoader(test_set, batch_size=32, shuffle=False)


            # create model and load weights from checkpoint
            model = Classifier().to(device)
            model.load_state_dict(torch.load(model_path), strict=False)

            model.eval() # set the model to evaluation mode
            with torch.no_grad():
            
                days = 0
                for test_data in test_loader:
                    inputs = test_data
                    inputs = inputs.to(device)
                    outputs = nn.Softmax(dim=1)(model(inputs))
                    
                    test_value, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
                        
                    for j in range(len(outputs)):
                        predicted_value[d][days + j].append(test_value.cpu().numpy()[j])
                        predicted_label[d][days + j].append(test_pred.cpu().numpy()[j])
                            
                    days += len(outputs)



    for d in [10, 20, 30, 40, 60, 120]:

        st.title(f'{d} days holding period')

        date_index = np.where(date_series == current_date)[0][0]

        predicted_value_n = predicted_value[d][date_index]
        predicted_label_n = predicted_label[d][date_index]


        sorted_value = []
        for i, value in enumerate(predicted_value_n):
            sorted_value.append((value, i))
        sorted_value = sorted(sorted_value)
        
        long_count = 0
        short_count = 0

        long_list = []
        short_list = []

        for value, idx in reversed(sorted_value):
           
            if predicted_label_n[idx] == 2 and long_count < 10: # long this stock

                fluaction_rate = code_para_map[codes[idx]][f'{d}d'][0] - 1
                long_list.append([f'{stocks[idx]} ({codes[idx]})', value, 
                                  f'{round(fluaction_rate * 100, 1)}%'])

                long_count += 1

            elif predicted_label_n[idx] == 0 and short_count < 10: # short this stock

                fluaction_rate =  code_para_map[codes[idx]][f'{d}d'][1] - 1
                short_list.append([f'{stocks[idx]} ({codes[idx]})', value, 
                                   f'{round(fluaction_rate * 100, 1)}%'])
    
                short_count += 1

            if long_count >= 10 and short_count >= 10:
                break

        st.subheader('Long')
        st.write(pd.DataFrame(long_list , columns = ['Stock', 'Softmax Value', 'Predicted Fluctuation Rate']))
        st.subheader('Short')
        st.write(pd.DataFrame(short_list , columns = ['Stock', 'Softmax Value', 'Predicted Fluctuation Rate']))
        
if __name__ == '__main__':
    main()
