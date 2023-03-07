
import pandas as pd
import json
import numpy as np
import os
import io
import ta


def make_technical_dataset(sym, df):
   """ """
   df_obs = pd.DataFrame()             # observation
   df_exch = pd.DataFrame()            # exchange; for order match

   df_exch = pd.concat([df_exch, df['close'].rename(sym)], axis=1)
   df = df[['open', 'high', 'low', 'close', 'volume']]

   df.columns = [f'{sym}:{c.lower()}' for c in df.columns]


   macd = ta.trend.MACD(close=df[f'{sym}:close'])
   df[f'{sym}:macd'] = macd.macd()
   df[f'{sym}:macd_diff'] = macd.macd_diff()
   df[f'{sym}:macd_signal'] = macd.macd_signal()

   rsi = ta.momentum.RSIIndicator(close=df[f'{sym}:close'])
   df[f'{sym}:rsi'] = rsi.rsi()

   bb = ta.volatility.BollingerBands(close=df[f'{sym}:close'], window=20, window_dev=2)
   df[f'{sym}:bb_bbm'] = bb.bollinger_mavg()
   df[f'{sym}:bb_bbh'] = bb.bollinger_hband()
   df[f'{sym}:bb_bbl'] = bb.bollinger_lband()

   atr = ta.volatility.AverageTrueRange(high=df[f'{sym}:high'], low=df[f'{sym}:low'], close=df[f'{sym}:close'])
   df[f'{sym}:atr'] = atr.average_true_range()


   df_obs = pd.concat([df_obs, df], axis=1)

   print(df_obs, df_exch)
   return df_obs.join(df_exch)



def main():
    """ """
    # make_technical_dataset(sym, data)
    """
    amzn =  pd.read_json("AMZNdaily.json")
    amzn = make_technical_dataset("amzn", amzn)
    amzn.to_csv('amzn_tech.csv')

    aapl =  pd.read_json("AAPLdaily.json")
    aapl = make_technical_dataset("aapl", aapl)
    aapl.to_csv('aapl_tech.csv')

    gme =  pd.read_json("GMEdaily.json")
    gme = make_technical_dataset("gme", gme)
    gme.to_csv('gme_tech.csv')
    """
    btc =  pd.read_csv("BTC-USD_1.csv", index_col=0)
    print(btc)
    btc = make_technical_dataset("btc", btc)
    btc.to_csv('btc_tech.csv')


if __name__ == '__main__':
    main()
