import pandas as pd
import math
import json
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
    signal[-(max_period-1):] = np.nan

    ret = pd.DataFrame({'triple_barrier_profit':p, 'triple_barrier_sell_time':t, 'triple_barrier_signal':signal})

    return ret

def data_generator(codes):

    date_serie = pd.read_csv('./data/indicator_data/close.csv')['date']

    close_df = pd.read_csv('./data/indicator_data/close.csv')
    roc_10_df = pd.read_csv('./data/indicator_data/roc_10.csv')
    roc_30_df = pd.read_csv('./data/indicator_data/roc_30.csv')
    cci_14_df = pd.read_csv('./data/indicator_data/cci_14.csv')
    cci_42_df = pd.read_csv('./data/indicator_data/cci_42.csv')
    adx_14_df = pd.read_csv('./data/indicator_data/adx_14.csv')
    adx_42_df = pd.read_csv('./data/indicator_data/adx_42.csv')
    atr_14_df = pd.read_csv('./data/indicator_data/atr_14.csv')
    atr_42_df = pd.read_csv('./data/indicator_data/atr_42.csv')
    ema_20_df = pd.read_csv('./data/indicator_data/ema_20.csv')
    ema_60_df = pd.read_csv('./data/indicator_data/ema_60.csv')
    sma_20_df = pd.read_csv('./data/indicator_data/sma_20.csv')
    sma_60_df = pd.read_csv('./data/indicator_data/sma_60.csv')
    willr_14_df = pd.read_csv('./data/indicator_data/willr_14.csv')
    willr_42_df = pd.read_csv('./data/indicator_data/willr_42.csv')
    dif_df = pd.read_csv('./data/indicator_data/dif.csv')
    macd_df = pd.read_csv('./data/indicator_data/macd.csv')
    k_9_df = pd.read_csv('./data/indicator_data/k_9.csv')
    d_9_df = pd.read_csv('./data/indicator_data/d_9.csv')
    k_27_df = pd.read_csv('./data/indicator_data/k_27.csv')
    d_27_df = pd.read_csv('./data/indicator_data/d_27.csv')
    rsi_14_df = pd.read_csv('./data/indicator_data/rsi_14.csv')
    rsi_42_df = pd.read_csv('./data/indicator_data/rsi_42.csv')


    with open('trend_params_300.json') as jsonfile:
        code_para_map = json.load(jsonfile)

    for i, code in enumerate(tqdm(codes)):
        
        close_serie = close_df[code].rename('close')
        series = [date_serie, close_serie]
        series.append(roc_10_df[code].rename('roc_10'))
        series.append(roc_30_df[code].rename('roc_30'))
        series.append(cci_14_df[code].rename('cci_14'))
        series.append(cci_42_df[code].rename('cci_42'))
        series.append(adx_14_df[code].rename('adx_14'))
        series.append(adx_42_df[code].rename('adx_42'))
        series.append(atr_14_df[code].rename('atr_14'))
        series.append(atr_42_df[code].rename('atr_42'))
        series.append(ema_20_df[code].rename('ema_20'))
        series.append(ema_60_df[code].rename('ema_60'))
        series.append(sma_20_df[code].rename('sma_20'))
        series.append(sma_60_df[code].rename('sma_60'))
        series.append(willr_14_df[code].rename('willr_14'))
        series.append(willr_42_df[code].rename('willr_42'))
        series.append(macd_df[code].rename('macd'))
        series.append(dif_df[code].rename('dif'))
        series.append(k_9_df[code].rename('k_9'))
        series.append(d_9_df[code].rename('d_9'))
        series.append(k_27_df[code].rename('k_27'))
        series.append(d_27_df[code].rename('d_27'))
        series.append(rsi_14_df[code].rename('rsi_14'))
        series.append(rsi_42_df[code].rename('rsi_42'))

        params = code_para_map[code]

        series.append(triple_barrier(close_serie, params['10d'][0], params['10d'][1], 11)['triple_barrier_signal'].rename('10d'))
        series.append(triple_barrier(close_serie, params['20d'][0], params['20d'][1], 21)['triple_barrier_signal'].rename('20d'))
        series.append(triple_barrier(close_serie, params['30d'][0], params['30d'][1], 31)['triple_barrier_signal'].rename('30d'))
        series.append(triple_barrier(close_serie, params['40d'][0], params['40d'][1], 41)['triple_barrier_signal'].rename('40d'))
        series.append(triple_barrier(close_serie, params['60d'][0], params['60d'][1], 61)['triple_barrier_signal'].rename('60d'))
        series.append(triple_barrier(close_serie, params['120d'][0], params['120d'][1], 121)['triple_barrier_signal'].rename('120d'))
        df = pd.concat(series, axis=1)

 


        df.to_csv(f'./data/training_data_300/raw_data_{i+1}.csv')


if __name__ == '__main__':
    with open('codes_300.txt', 'r') as f:
        codes = f.read().split()

    data_generator(codes)
