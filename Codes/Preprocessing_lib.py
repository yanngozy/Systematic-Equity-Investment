import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_benchmark_returns(df):
    """
        Compute the benchmark.
    """

    def group_returns(group):
    
        if "MarketCap" in group.columns and group["MarketCap"].notna().sum() > 0:
            weights_col = "MarketCap"
        else:
            weights_col = "Close"
            

        valid_mask = (
            group["returns"].notna() & 
            group[weights_col].notna() & 
            (group[weights_col] > 0)
        )
        

        if not valid_mask.any():
            return np.nan
            
        sub_group = group[valid_mask]
        
        return np.average(sub_group["returns"], weights=sub_group[weights_col])


    sp = (
        df.groupby("Date", group_keys=False)
        .apply(group_returns)
        .rename("benchmark")
        .reset_index()
    )

    features_df = df.merge(sp, on="Date", how="left")

    return features_df


def compute_sectional_zscores(df: pd.DataFrame, columns: list, eps: float = 1e-9) -> pd.DataFrame:
    """
        Standardize (Z-Score)  columns per Date and per sector 
        Utilise la vectorisation native de Pandas au lieu des boucles lambda/scipy.
    """
    
    groupby_keys = ["Date", "sector"]
    
    
    for col in columns:
        if col not in df.columns:
            continue
            
        mean = df.groupby(groupby_keys)[col].transform("mean")
        std  = df.groupby(groupby_keys)[col].transform("std")
        
        if col=='returns':
            df['returns_zscore'] = (df[col] - mean) / (std + eps)   
        else:  
            df[col] = (df[col] - mean) / (std + eps)
        
    return 0


def compute_factor_signals(data: pd.DataFrame , eps=10**(-9)) :
    """
        Compute raw profitability ratios.
        Input  : panel df with columns listed above
        Output : same df with new signal columns appended
    """

    df = data.copy()
    df = compute_benchmark_returns(df)

    # ---------- Quality / Profitability -------
    #df['gross_margin']     =  df['GrossProfit']     / (df['Revenue'] + eps)
    df['operating_margin']  =  df['OperatingIncome']  / (df['Revenue'] + eps)
    df['net_margin']        =  df['NetIncome']        / (df['Revenue'] + eps)
    df["accruals"]          =  (df["NetIncome"] - df["OperatingCF"]) / (df["TotalAssets"] + eps)
    df['roa']               =  df['NetIncome'] / (df['TotalAssets'] + eps)
    df['roe']               =  df['NetIncome'] / (df['Equity'] + eps)
    df['debt_ratio']        =  df["LongTermDebt"] / (df["TotalAssets"] + eps)
    df["asset_turnover"]    =  df["Revenue"] / (df["TotalAssets"] + eps)

    # ROIC proxy: operating income / (assets - current liabilities)
    if "TotalLiabilities" in df.columns:
        invested_capital = df["TotalAssets"] - df["TotalLiabilities"]
        safe_ic = invested_capital.where(invested_capital > 0, np.nan)
        df["roic_proxy"] = df["OperatingIncome"] / (safe_ic + eps)
    else:
        df["roic_proxy"] = df["OperatingIncome"] / (df["TotalAssets"] + eps)



    # --------------- Value ---------------
    # E/P Ratio (Earnings Yield) 
   
    df["book_to_market"]  =  df["Equity"]  / (df["MarketCap"] + eps)
    df["earnings_yield"]  =  df["NetIncome"] / (df["MarketCap"] + eps)
    df["sales_yield"]     =  df["Revenue"] / (df["MarketCap"] + eps)

    safe_close            =  df["Close"].where(df["Close"] > 0, np.nan)
    df["eps_yield"]       =  df["EPS_Diluted"] / (safe_close + eps)
    df["eps_yield_basic"] =  df["EPS_Basic"] / (safe_close + eps)
    
    # B/P Ratio (Book-to-Price) 
    if "SharesOutstanding" in df.columns:
        bvps                 =  df["Equity"] / (df["SharesOutstanding"] + eps)      # Book Value Per Share = Equity / SharesOutstanding
        df["book_to_price"]  =  bvps / (safe_close + eps)   
        df["sales_to_price"] =  (df["Revenue"] / (df["SharesOutstanding"] + eps)) / (safe_close + eps)
    else:
        df["book_to_price"]  =  np.nan
        df["sales_to_price"] =  np.nan

   
    return df



def build_technical_features(df: pd.DataFrame, eps = 10**(-9)) -> pd.DataFrame:
    """
        Compute technical indicators from OHLCV + returns panel data.
    
        Required columns : Date, Ticker, Open, High, Low, Close, Volume, returns
        Returns          : original df + indicator columns (+ _z z-scored variants)
    """
   
    
    # ────────────────────────────────── Trend ──────────────────────────────────────────
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    dg = df[['Date','returns']].copy()
    g  = df.groupby('Ticker')
    
    for w in [5,12, 26, 50]:
        dg[f'ema_{w}'] = g['Close'].transform(lambda x: x.ewm(span=w, adjust=False).mean())
        df[f'close_vs_ema_{w}']   = (df['Close'] - dg[f'ema_{w}'])  / (dg[f'ema_{w}'] + eps)

    df['macd']        = dg['ema_12'] - dg['ema_26']
    df['macd_signal'] = g['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df['macd_hist']   = df['macd'] - df['macd_signal']
  
    
    # ────────────────────────── Momentum ─────────────────────────────────────────────
    
    def _rsi(s, p=14):
        d = s.diff()
        g = d.clip(lower=0).ewm(alpha=1/p, adjust=False).mean()
        l = (-d.clip(upper=0)).ewm(alpha=1/p, adjust=False).mean()
        return 100 - 100 / (1 + g / l.replace(0, np.nan))

    df['rsi_14'] = g['Close'].transform(_rsi)
    df['rsi_7']  = g['Close'].transform(lambda x: _rsi(x, 7))
    for w in [5, 10, 20, 60]:
        df[f'roc_{w}'] = g['Close'].transform(lambda x: x.pct_change(w))

    def _stoch(grp, k=14):
        lo, hi = grp['Low'].rolling(k).min(), grp['High'].rolling(k).max()
        return 100 * (grp['Close'] - lo) / (hi - lo).replace(0, np.nan)

    df['stoch_k'] = df.groupby('Ticker', group_keys=False).apply(_stoch)
    df['stoch_d'] = g['stoch_k'].transform(lambda x: x.rolling(3).mean())

    def _williams_r(grp, w=14):
        hi, lo = grp['High'].rolling(w).max(), grp['Low'].rolling(w).min()
        return -100 * (hi - grp['Close']) / (hi - lo).replace(0, np.nan)

    df['williams_r'] = df.groupby('Ticker', group_keys=False).apply(_williams_r)

    for w in [5,21,63,126,252]:
        df[f"mom_{w}"] = df.groupby("Ticker")["Close"].pct_change(w)
    
    
    # ───────────────────────────── Volatility ────────────────────────────────────────────────

    sma20 = g['Close'].transform(lambda x: x.rolling(20).mean())
    std20 = g['Close'].transform(lambda x: x.rolling(20).std())
    df['sharpe']   = sma20/(std20 + eps)
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (sma20 + eps)
    df['bb_pct']   = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + eps)

    def _atr(grp, w=14):
        tr = pd.concat([
            grp['High'] - grp['Low'],
            (grp['High'] - grp['Close'].shift()).abs(),
            (grp['Low']  - grp['Close'].shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=w, adjust=False).mean()

    df['atr_14']  = df.groupby('Ticker', group_keys=False).apply(_atr)
    df['atr_pct'] = df['atr_14'] / df['Close']

    for w in [10, 20, 60]:
        df[f'hvol_{w}'] = g['returns'].transform(lambda x: x.rolling(w).std() * np.sqrt(252))

    df['vol_ratio'] = df['hvol_10'] / (df['hvol_60'] + eps)

    ema20 = g['Close'].transform(lambda x: x.ewm(span=20).mean())
    df['kc_upper'] = ema20 + 2 * df['atr_14']
    df['kc_lower'] = ema20 - 2 * df['atr_14']


    # ──────────────────────────── Volume ────────────────────────────────────────────────────────

    def _obv(grp):
        return (np.sign(grp['Close'].diff()).fillna(0) * grp['Volume']).cumsum()

    df['obv'] = df.groupby('Ticker', group_keys=False).apply(_obv)

    def _vwap(grp, w=20):
        tp = (grp['High'] + grp['Low'] + grp['Close']) / 3
        return (tp * grp['Volume']).rolling(w).sum() / grp['Volume'].rolling(w).sum()

    df['vwap_20']      = df.groupby('Ticker', group_keys=False).apply(_vwap)
    df['close_vs_vwap'] = (df['Close'] - df['vwap_20']) / df['vwap_20']

    def _cmf(grp, w=20):
        hl  = (grp['High'] - grp['Low']).replace(0, np.nan)
        mfv = ((grp['Close'] - grp['Low']) - (grp['High'] - grp['Close'])) / hl * grp['Volume']
        return mfv.rolling(w).sum() / grp['Volume'].rolling(w).sum()

    df['cmf_20'] = df.groupby('Ticker', group_keys=False).apply(_cmf)

    def _mfi(grp, w=14):
        tp  = (grp['High'] + grp['Low'] + grp['Close']) / 3
        mf  = tp * grp['Volume']
        pos = mf.where(tp > tp.shift(), 0).rolling(w).sum()
        neg = mf.where(tp < tp.shift(), 0).rolling(w).sum()
        return 100 - 100 / (1 + pos / neg.replace(0, np.nan))

    df['mfi_14']      = df.groupby('Ticker', group_keys=False).apply(_mfi)
    vol_ma = g['Volume'].transform(lambda x: x.rolling(20).mean())
    df['volume_surge'] = df['Volume'] / vol_ma

    # ──────────── Structure ────────────────────────────────────────────────────────────
    df['high_52w']      = g['High'].transform(lambda x: x.rolling(252).max())
    df['low_52w']       = g['Low'].transform(lambda x: x.rolling(252).min())
    df['pct_from_high'] = (df['Close'] - df['high_52w']) / df['high_52w']
    df['pct_from_low']  = (df['Close'] - df['low_52w'])  / df['low_52w']
    df['gap']           = g['Open'].transform(lambda x: (x - x.shift()) / x.shift())
    df['body_size']     = (df['Close'] - df['Open']).abs() / df['Open']
    df['upper_wick']    = (df['High'] - df[['Open','Close']].max(axis=1)) / df['Open']
    df['lower_wick']    = (df[['Open','Close']].min(axis=1) - df['Low']) / df['Open']
    df['bullish_bar']   = (df['Close'] - df['Open']) / df['Open']
    df["hl_spread"]     = (df["High"] - df["Low"]) / ( df["Close"] + eps )

    return df


