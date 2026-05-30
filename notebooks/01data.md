```python
%run 00setup.ipynb
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload
    âœ… Setup complete
    


```python
tickers = ["SPY","QQQ","DIA","IWM","VUG","VTV","USMV","FEZ","EWU","EWJ","EEM",
           "TLT","IEF","SHY","LQD","GLD","DBC","VNQ","XLU","XLP"]

hist = yf.download(tickers, start="2010-01-01", end="2025-12-31", auto_adjust=False, progress=False)
hist.columns = hist.columns.set_names(["field","ticker"])
df = (
    hist.stack("ticker", future_stack=True)   # index now: date, ticker
        .reset_index()                        # columns: date, ticker, fields...
        .rename(columns={
            "Date":"date","Open":"open","High":"high","Low":"low",
            "Close":"close","Adj Close":"adj_close","Volume":"volume"
        })
        [["date","ticker","open","high","low","close","adj_close","volume"]]
        .sort_values(["ticker","date"])
        .reset_index(drop=True)
)
```


```python
wide_adj = df.pivot(index="date", columns="ticker", values="adj_close").dropna(how="any")
common_dates = wide_adj.index

clean = (df[df["date"].isin(common_dates)]
        .query("volume > 0")
        .sort_values(["ticker","date"])
        .reset_index(drop=True))

save_df(clean, "prices_clean", DATA_DIR)
print("âœ… Saved:",
      len(clean), "Rows |",
      clean['ticker'].nunique(), "Tickers |",
      "Dates:", common_dates.min().date(), "â†’", common_dates.max().date())
```

    âœ… Saved: 71373 Rows | 20 Tickers | Dates: 2011-10-20 â†’ 2025-12-30
    
