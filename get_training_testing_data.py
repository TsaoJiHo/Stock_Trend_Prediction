import pandas as pd
import math
import numpy as np
from tqdm import tqdm

def triple_barrier(price, ub, lb, max_period):

    def end_price(s):
        return np.append(s[(s / s[0] > ub) | (s / s[0] < lb)], s[-1])[0]/s[0]
    
    r = np.array(range(max_period))
    
    def end_time(s):
        return np.append(r[(s / s[0] > ub) | (s / s[0] < lb)], max_period-1)[0]

    p = price.rolling(max_period).apply(end_price, raw=True).shift(-max_period+1)
    t = price.rolling(max_period).apply(end_time, raw=True).shift(-max_period+1)
    t = pd.Series([t.index[int(k+i)] if not math.isnan(k+i) else np.datetime64('NaT') 
                   for i, k in enumerate(t)], index=t.index).dropna()

    signal = pd.Series(1, p.index)
    signal.loc[p > ub] = 2
    signal.loc[p < lb] = 0
    ret = pd.DataFrame({'triple_barrier_profit':p, 'triple_barrier_sell_time':t, 'triple_barrier_signal':signal})

    return ret

def data_generator(codes):
    training_data = []
    testing_data = []

    date_serie = pd.read_csv('./indicator_data/close.csv')['date']

    close_df = pd.read_csv('./indicator_data/close.csv')
    k_df = pd.read_csv('./indicator_data/k.csv')
    d_df = pd.read_csv('./indicator_data/d.csv')
    rsi_7_df = pd.read_csv('./indicator_data/rsi_7.csv')
    rsi_14_df = pd.read_csv('./indicator_data/rsi_14.csv')
    dif_df = pd.read_csv('./indicator_data/dif.csv')
    macd_df = pd.read_csv('./indicator_data/macd.csv')
    willr_df = pd.read_csv('./indicator_data/willr.csv')

    for code in tqdm(codes):
     
        close_serie = close_df[code].rename('close')

        k_serie = k_df[code].rename('k')
        d_serie = d_df[code].rename('d')
        rsi_7_serie = rsi_7_df[code].rename('rsi_7')
        rsi_14_serie = rsi_14_df[code].rename('rsi_14')
        macd_serie = macd_df[code].rename('macd')
        dif_serie = dif_df[code].rename('dif')
        willr_serie = willr_df[code].rename('willr')

        delta_k_serie = (k_serie - k_serie.shift(1)).rename('delta_k')
        delta_d_serie = (d_serie - d_serie.shift(1)).rename('delta_d')
        delta_rsi_serie = (rsi_7_serie - rsi_7_serie.shift(1)).rename('delta_rsi_7')
        delta_macd_serie = (macd_serie - macd_serie.shift(1)).rename('delta_macd')
        delta_dif_serie = (dif_serie - dif_serie.shift(1)).rename('delta_dif')
        delta_willr_serie = (willr_serie - willr_serie.shift(1)).rename('delta_willr')

        # change = (close_serie - close_serie.shift(1)).shift(-1).rename('label')
        # label_serie = (change > 0).astype('int')[:-1]
        label_serie_30 = triple_barrier(close_serie, 1.1, 0.9, 31)['triple_barrier_signal'].rename('30d')
        label_serie_15 = triple_barrier(close_serie, 1.07, 0.93, 16)['triple_barrier_signal'].rename('15d')
        label_serie_7 = triple_barrier(close_serie, 1.04, 0.96, 8)['triple_barrier_signal'].rename('7d')
        label_serie_3 = triple_barrier(close_serie, 1.02, 0.98, 4)['triple_barrier_signal'].rename('3d')
        label_serie_1 = triple_barrier(close_serie, 1.01, 0.99, 2)['triple_barrier_signal'].rename('1d')
        df = pd.concat([date_serie, close_serie, k_serie, d_serie, rsi_7_serie,  rsi_14_serie,
                        dif_serie, macd_serie, willr_serie,
                        label_serie_1, label_serie_3, label_serie_7, label_serie_15, label_serie_30], axis=1)
        # df = pd.concat([date_serie, close_serie, k_serie, d_serie, rsi_7_serie,  rsi_14_serie,
        #                 dif_serie, macd_serie, willr_serie,
        #                 label_serie_30], axis=1)
        # df = pd.concat([date_serie, close_serie, k_serie, 
        #                  d_serie,  rsi_serie, macd_serie, 
        #                   willr_serie, label_serie], axis=1)
        df = df.dropna()
        # indexNames = df[ df['30d'] == -1 ].index
        # Delete these row indexes from dataFrame
        # df.drop(indexNames , inplace=True)
        
        split_gap = int(len(df) * 0.8)
        training_data.append(df[:split_gap])
        testing_data.append(df[split_gap:])

    pd.concat(training_data).to_csv('./training_data_v2.csv')
    pd.concat(testing_data).to_csv('./testing_data_v2.csv')

if __name__ == '__main__':
    with open('codes.txt', 'r') as f:
        codes = f.read().split()

    data_generator(codes)
