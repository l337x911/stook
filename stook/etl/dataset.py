from pathlib import Path
import vectorbt as vbt
import pandas as pd
import pytz
from datetime import datetime, timedelta
from dateutil.parser import parse
from copy import deepcopy
from tqdm import tqdm
import itertools
import dateparser
import os
import logging
import vectorbt as vbt
from vectorbt import _typing as tp

DATASET_PATH = os.environ.get("DATASET_PATH", "data")

class StooqData(vbt.Data):
    """Extracts ohlcv data from text files downloaded from stooq.com/db/h."""

    @staticmethod
    def filter_valid_symbols(symbols: tp.List[tp.Label]):
        """Check symbols are present in data folder."""
        stock_root = f"{DATASET_PATH}/d_us_txt/data/daily/us"
        missing = []
        valids = []
        for symbol in symbols:
            stock_paths = [n for n in Path(stock_root).glob(f"**/{symbol}.*txt")]
            if len(stock_paths) == 0:
                missing.append(symbol)
                continue
            valids.append(symbol)
        return missing, valids

    @classmethod
    def download_symbol(cls, symbol: tp.Label, start: tp.Optional[tp.DatetimeLike] = None, end: tp.Optional[tp.DatetimeLike] = None) -> tp.Frame:
        """Loads a symbol's open, high, low, close, and volume on a given time range as a pandas dataframe."""
        stock_root = f"{DATASET_PATH}/d_us_txt/data/daily/us"
        stock_paths = [n for n in Path(stock_root).glob(f"**/{symbol}.*txt")]
        data = []
        if len(stock_paths) == 0:
            raise ValueError(f"no {symbol} data found.")
        for n in stock_paths:
            #print(n)
            data.append(pd.read_csv(str(n), sep=','))
        data = pd.concat(data)
        data['<DATE>'] = pd.to_datetime(data['<DATE>'], format='%Y%m%d')

        start_dt = pd.to_datetime(start, format='%Y-%m-%d')
        end_dt = pd.to_datetime(end, format='%Y-%m-%d')

        ohlcv = data.loc[(data['<DATE>'] >= start_dt) & (data['<DATE>'] <= end_dt),:]\
                .sort_values(by='<DATE>')\
                .rename(columns={'<DATE>':'date', '<OPEN>':'open', '<CLOSE>':'close', '<VOL>':'volume', '<HIGH>':'high', '<LOW>':'low'})
        return ohlcv.set_index('date').loc[:,['open', 'high', 'low', 'close', 'volume']]

