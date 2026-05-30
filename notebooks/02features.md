```python
%run 00setup.ipynb
```

    âœ… Setup complete
    


```python
data = load_df("prices_clean",DATA_DIR,fmt="parquet")
data = data.sort_values(["ticker","date"]).reset_index(drop=True)
h = 3
data["target_raw"] = data.groupby("ticker")["adj_close"].shift(-h) / data["adj_close"] - 1.0
```


```python


def cs_madscore(x):
    x = x.astype(float)
    valid = x.dropna()

    if valid.empty:
        return pd.Series(np.nan, index=x.index)

    med = valid.median()
    mad = np.median(np.abs(valid - med))

    if mad < 1e-12:
        return pd.Series(0.0, index=x.index)

    return (x - med) / (1.4826 * mad)

# cross-sectional robust normalization by date
data["target"] = (
    data.groupby("date")["target_raw"]
        .transform(cs_madscore)
        .clip(-3, 3)
)

# drop rows where future return cannot be computed
data = data.dropna(subset=["target_raw", "target"])
```


```python
# Momentum
data["momentum_5"]  = data.groupby("ticker")["adj_close"].pct_change(5)
data["momentum_10"]  = data.groupby("ticker")["adj_close"].pct_change(10)
data["momentum_25"]  = data.groupby("ticker")["adj_close"].pct_change(25)
data["momentum_100"] = data.groupby("ticker")["adj_close"].pct_change(100)
```


```python
#SMA
data["sma20"]     = data.groupby("ticker")["adj_close"].transform(lambda x: x.rolling(20).mean())
data["sma20_dev"] = data["adj_close"] / data["sma20"] - 1
```


```python
#RSI
ret = data.groupby("ticker")["adj_close"].pct_change()
gain = ret.clip(lower=0)
loss = -ret.clip(upper=0)
avg_gain_14 = gain.groupby(data["ticker"]).transform(lambda x: x.rolling(14).mean())
avg_loss_14 = loss.groupby(data["ticker"]).transform(lambda x: x.rolling(14).mean())
avg_gain_7 = gain.groupby(data["ticker"]).transform(lambda x: x.rolling(7).mean())
avg_loss_7 = loss.groupby(data["ticker"]).transform(lambda x: x.rolling(7).mean())
rsi_14 = avg_gain_14 / avg_loss_14
rsi_7 = avg_gain_7 / avg_loss_7
data["rsi14"] = 100 - (100 / (1 + rsi_14))
data["rsi7"] = 100 - (100 / (1 + rsi_7))
```


```python
#Volatility
data["volatility5"]  = ret.groupby(data["ticker"]).transform(lambda x: x.rolling(5).std())
data["volatility20"]  = ret.groupby(data["ticker"]).transform(lambda x: x.rolling(20).std())
data["volatility100"] = ret.groupby(data["ticker"]).transform(lambda x: x.rolling(100).std())
data["volatility100_ratio"] = data["volatility20"] / data["volatility100"]
data["volatility20_ratio"] = data["volatility5"] / data["volatility20"]
```


```python
#ATR
prev_close = data.groupby("ticker")["close"].shift(1)
high_low   = data["high"] - data["low"]
high_close = (data["high"] - prev_close).abs()
low_close  = (data["low"] - prev_close).abs()
t = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

data["atr14"] = t.groupby(data["ticker"]).transform(lambda x: x.rolling(14).mean())
data["atr5"] = t.groupby(data["ticker"]).transform(lambda x: x.rolling(5).mean())
atr10  = t.groupby(data["ticker"]).transform(lambda x: x.rolling(10).mean())
atr100 = t.groupby(data["ticker"]).transform(lambda x: x.rolling(100).mean())
data["delta_atr10_atr100"] = atr10 / atr100 - 1
```


```python
data["log_vol"]   = np.log1p(data["volume"])
data["vol_std50"] = data.groupby("ticker")["volume"].transform(lambda x: x.rolling(50).std())
data["vol_slope"] = data.groupby("ticker")["volume"].transform(
lambda x: x.rolling(50).apply(lambda w: (w[-1] - w[0]) / 50, raw=True))
```


```python
# Bollinger Bands
data["ma50"] = data.groupby("ticker")["adj_close"].transform(lambda x: x.rolling(50).mean())
data["std50"] = data.groupby("ticker")["adj_close"].transform(lambda x: x.rolling(50).std())
data["bb_lower"] = data["ma50"] - 2 * data["std50"]
data["bb_upper"] = data["ma50"] + 2 * data["std50"]

data["bb50_lower_slope"] = data.groupby("ticker")["bb_lower"].transform(
    lambda x: x.rolling(50).apply(lambda w: (w[-1] - w[0]) / 50, raw=True)
)
data["bb50_upper_slope"] = data.groupby("ticker")["bb_upper"].transform(
    lambda x: x.rolling(50).apply(lambda w: (w[-1] - w[0]) / 50, raw=True)
)
```


```python
#z-score of 20d momentum
mm20 = data.groupby("ticker")["adj_close"].pct_change(20)
data["zsc_ret20"] = mm20.groupby(data["date"]).transform(lambda x: (x - x.mean()) / x.std(ddof=0))
```


```python
feature_cols = [
    "momentum_5","momentum_10","momentum_25","momentum_100",
    "sma20_dev","rsi7","rsi14","volatility5","volatility20","volatility100",
    "volatility20_ratio","volatility100_ratio","atr5","atr14",
    "delta_atr10_atr100","log_vol","vol_std50","vol_slope",
    "bb50_lower_slope","bb50_upper_slope","zsc_ret20",
    "target_raw","target"
]

data_features = (
    data[["date","ticker"] + feature_cols]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .reset_index(drop=True)
)

save_df(data_features, "features", DATA_DIR)
```
