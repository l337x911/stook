import os
import urllib.request
from urllib.error import HTTPError
import argparse
import logging
from pathlib import Path
import vectorbt as vbt 
from datetime import datetime, timedelta
import pandas as pd
pd.options.mode.chained_assignment = None
from numba import njit

import numpy as np
from stook.etl.dataset import DATASET_PATH, StooqData


def create_rand_hold_topk_orders(close_prices, top_k=3, init_cash=1000, approx_triggers_per_year=4, seed=42):
    """Creates random intervals to rank stocks by best performance over the interval, select top k to order.
    If new stock goes into top k, then exit dropped stock and rebalance remaining cash.
    """
    rnd_gen = np.random.default_rng(seed)
    num_rows = close_prices.shape[0]
    
    rnd_trigger_lengths = rnd_gen.poisson(252 / approx_triggers_per_year, close_prices.shape[0])
    trigger_indices = np.cumsum(rnd_trigger_lengths)
    trigger_indices = trigger_indices[trigger_indices < num_rows]

    trigger_close_prices = close_prices.iloc[trigger_indices[1:],:]
    price_change = trigger_close_prices / close_prices.iloc[trigger_indices[:-1],:].values
    price_change = price_change.fillna(-100)
    
    ranked = pd.DataFrame(np.argsort(-price_change, axis=1), price_change.index, columns=price_change.columns)
    top_k_col = ranked < top_k

    sizes = pd.DataFrame(np.zeros_like(price_change.values), index=ranked.index, columns=ranked.columns)
    for i, trigger_day in enumerate(sizes.index):
        if i == 0:
            # init entry orders
            selected = top_k_col.loc[trigger_day,:]
            sizes.loc[trigger_day,selected] = init_cash / top_k / trigger_close_prices.loc[trigger_day,selected]
        else:
            # add exits from previous, and new entries
            prev = sizes.index[ i - 1 ]
            need_to_exit = top_k_col.loc[trigger_day,:].values > top_k_col.loc[prev,:].values
            sizes.loc[trigger_day, need_to_exit] = -sizes.loc[:trigger_day, need_to_exit].sum()

            cash_on_hand = sum(sizes.loc[:trigger_day, need_to_exit].sum() * trigger_close_prices.loc[trigger_day, need_to_exit])
    
            need_to_enter = top_k_col.loc[trigger_day,:].values > top_k_col.loc[prev,:].values
            sizes.loc[trigger_day,need_to_enter] = cash_on_hand / sum(need_to_enter) / trigger_close_prices.loc[trigger_day,need_to_enter]            

    print(close_prices.iloc[:6,:].to_csv(None, sep='\t', float_format='%.2f'))
    print(price_change.iloc[:5,:].to_csv(None, sep='\t', float_format='%.2f'))
    print(ranked.iloc[:5,:].to_csv(None, sep='\t'))
    print(sizes.iloc[:5,:].to_csv(None, sep='\t', float_format='%.2f'))
    
    return trigger_close_prices.fillna(0.001), sizes


def main(args):
    sp500_tickers = pd.read_csv(f"{DATASET_PATH}/sp_500_basics.csv")
    symbols = [str.lower(s) for s in sp500_tickers['Symbol']]
    symbols = symbols[:20]

    missing_symbols, symbols = StooqData.filter_valid_symbols(symbols)

    logging.info(f"tracking sp500 symbols: {len(symbols)} with {len(missing_symbols)} missing.") 
    stook_data = StooqData.download(symbols, start='2000-01-01', end='2025-04-01', missing_columns='drop')
    open_prices, high_prices, low_prices, close_prices, volumes = stook_data.get()

    init_cash = 1000
    price, size = create_rand_hold_topk_orders(close_prices, top_k=args.top_k, approx_triggers_per_year=4, init_cash=init_cash, seed=args.seed)
    logging.info(f"triggered {len(price)} times.")

    topk_pf = vbt.Portfolio.from_orders(price,
            size=size,
            fixed_fees=0.01)
    
    print(topk_pf.returns_stats(freq='d'))
    print(topk_pf.trades.records.to_csv(None, float_format='%.2f'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulate triggering switch at random intervals.")
    parser.add_argument("-k", dest='top_k', default=3, type=int, help='the number of stocks to hold and rebalance. default: 3') 
    #parser.add_argument("--lag", dest='rate_of_return_period', default=90, type=int, help='number of previous days to calculate rate of return. default: 90') 
    parser.add_argument("--seed", dest='seed', default=42, type=int, help='seed for random. default: 42') 

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(process)d:%(levelname)s:%(message)s")



    main(args)
