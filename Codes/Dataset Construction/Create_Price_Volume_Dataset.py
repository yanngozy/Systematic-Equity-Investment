import numpy as np
import pandas as pd 
import yfinance as yf
import time

# We use yahoo finance to extract price volume data for s&p 500 stocks 

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500 = pd.read_html(url, storage_options=headers)[0]

s_p_500_tickers = sp500["Symbol"].str.replace(".", "-", regex=False).tolist()


def generate_price_volume_dataset(ticker=s_p_500_tickers,start_date="2005-01-01", end_date="2026-03-31",batch_size=100)

    all_closes = []
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        data  = yf.download(batch, start=start_date, end=end_date,
                           auto_adjust=True, progress=False)
        data=data.stack(level="Ticker").reset_index(level='Ticker')
        all_closes.append(data)
        
    df = pd.concat(all_closes, axis=0)
    df = df.sort_values(["Date"])
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    df.reset_index(inplace=True)

    FIELDS = ["symbol", "shortName", "sector", "industry"]

    records = []
    
    for i, sym in enumerate(tickers):
        info = yf.Ticker(sym).info
        record = {f: info.get(f) for f in FIELDS}
        record["symbol"] = sym
        records.append(record)        
    
    df_fundamentals = pd.DataFrame(records)
    df_fundamentals = df_fundamentals[['symbol','shortName','sector','industry']].rename(
                                                            {'symbol':'Ticker'},
                                                            axis=1
                                                            )
    df = pd.merge(df,df_fundamentals,on='Ticker',how='left')
    
    return df
