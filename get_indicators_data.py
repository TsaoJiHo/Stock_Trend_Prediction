from finlab import data
# https://ai.finlab.tw/api_token/

data.get('price:收盤價').to_csv('./data/indicator_data/close.csv')

data.indicator('RSI', timeperiod=14).to_csv('./data/indicator_data/rsi_14.csv')
data.indicator('RSI', timeperiod=42).to_csv('./data/indicator_data/rsi_42.csv')


k_9, d_9 = data.indicator('STOCH', timeperiod=9)
k_9.to_csv('./data/indicator_data/k_9.csv')
d_9.to_csv('./data/indicator_data/d_9.csv')

k_27, d_27 = data.indicator('STOCH', timeperiod=27)
k_27.to_csv('./data/indicator_data/k_27.csv')
d_27.to_csv('./data/indicator_data/d_27.csv')

macd = data.indicator('MACD')
macd[0].to_csv('./data/indicator_data/dif.csv')
macd[1].to_csv('./data/indicator_data/macd.csv')

data.indicator('WILLR', timeperiod=14).to_csv('./data/indicator_data/willr_14.csv')
data.indicator('WILLR', timeperiod=42).to_csv('./data/indicator_data/willr_42.csv')

data.indicator('SMA', timeperiod=20).to_csv('./data/indicator_data/sma_20.csv')
data.indicator('SMA', timeperiod=60).to_csv('./data/indicator_data/sma_60.csv')

data.indicator('EMA', timeperiod=20).to_csv('./data/indicator_data/ema_20.csv')
data.indicator('EMA', timeperiod=60).to_csv('./data/indicator_data/ema_60.csv')

data.indicator('ATR', timeperiod=14).to_csv('./data/indicator_data/atr_14.csv')
data.indicator('ATR', timeperiod=42).to_csv('./data/indicator_data/atr_42.csv')

data.indicator('ADX', timeperiod=14).to_csv('./data/indicator_data/adx_14.csv')
data.indicator('ADX', timeperiod=42).to_csv('./data/indicator_data/adx_42.csv')

data.indicator('CCI', timeperiod=14).to_csv('./data/indicator_data/cci_14.csv')
data.indicator('CCI', timeperiod=42).to_csv('./data/indicator_data/cci_42.csv')

data.indicator('ROC', timeperiod=10).to_csv('./data/indicator_data/roc_10.csv')
data.indicator('ROC', timeperiod=30).to_csv('./data/indicator_data/roc_30.csv')
