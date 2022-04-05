from finlab import data
# https://ai.finlab.tw/api_token/

close = data.get('price:收盤價')
close.to_csv('./indicator_data/close.csv')

rsi_7 = data.indicator('RSI', timeperiod=7)
rsi_7.to_csv('./indicator_data/rsi_7.csv')
rsi_14 = data.indicator('RSI', timeperiod=14)
rsi_14.to_csv('./indicator_data/rsi_14.csv')


k, d = data.indicator('STOCH')
k.to_csv('./indicator_data/k.csv')
d.to_csv('./indicator_data/d.csv')

macd = data.indicator('MACD')
macd[0].to_csv('./indicator_data/dif.csv')
macd[1].to_csv('./indicator_data/macd.csv')

# ema_n = data.indicator('EMA', timeperiod=12)
# ema_m = data.indicator('EMA', timeperiod=26)
# dif = ema_n - ema_m
# dif.to_csv('./indicator_data/dif.csv')

willr = data.indicator('WILLR')
willr.to_csv('./indicator_data/willr.csv')
