import os
import urllib.request
from urllib.error import HTTPError
import argparse
import logging
import dill
from pathlib import Path
import vectorbt as vbt 
from datetime import datetime, timedelta
import pandas as pd
pd.options.mode.chained_assignment = None
from numba import njit

import numpy as np
from stook.etl.dataset import TRADING_DAYS_PER_YEAR, DATASET_PATH, StooqData


def create_rand_hold_topk_orders(rand_gen, close_prices, top_k=3, approx_triggers_per_year=4):
    """Creates random intervals to rank stocks by best performance over the interval, select top k to order.
    If new stock goes into top k, then exit dropped stock and rebalance remaining cash.

    :param rand_gen: random number generator.
    :param close_prices: End of day prices for assets.
    :param top_k: Number of assets to hold on at any given time.
    :param init_cash: Starting cash amount to distribute across portfolio assets.
    :param approx_triggers_per_year: expected number of triggers to eval/re-balance assets per year.
    """
    # set random trigger dates
    num_rows = close_prices.shape[0]
    rnd_trigger_lengths = rand_gen.poisson(TRADING_DAYS_PER_YEAR / approx_triggers_per_year, close_prices.shape[0])
    trigger_indices = np.cumsum(rnd_trigger_lengths)
    trigger_indices = list(trigger_indices[trigger_indices < num_rows])
    if trigger_indices[-1] != (num_rows - 1):
        trigger_indices.append(num_rows - 1)
    logging.debug("num of triggers: %s, triggers: %s", len(trigger_indices), trigger_indices)
    # rank stocks on trigger dates
    trigger_close_prices = close_prices.iloc[trigger_indices[1:],:]
    price_change = trigger_close_prices / close_prices.iloc[trigger_indices[:-1],:].values
    
    ranked = price_change.rank(axis=1, ascending=False, method='first', numeric_only=True)
    top_k_col = (ranked <= top_k) & ranked.notnull()
    logging.debug("ranked: %s", ranked.iloc[:5,:].to_csv(None, sep='\t'))
    
    sizes = pd.DataFrame(np.zeros_like(price_change.values), index=ranked.index, columns=ranked.columns)
    sizes.loc[:,:] = np.nan

    # init entry orders
    trigger_day = sizes.index[0]
    selected = top_k_col.loc[trigger_day,:]
    sizes.loc[trigger_day,selected] = 1 / top_k
    logging.debug("init_sizes: %s %s", sizes.shape, sizes.loc[trigger_day,selected])

    for i in range(1, sizes.shape[0] - 1):
        prev = sizes.index[i - 1]
        trigger_day = sizes.index[i]
        
        # add exits from previous sizes, and distribute cash to new entries
        prev_selected = top_k_col.loc[prev,:].values
        selected = top_k_col.loc[trigger_day,:].values
        logging.debug("prev:%s to_enter/keep: %s", sizes.columns[prev_selected], sizes.columns[selected])

        need_to_exit = selected < prev_selected
        exit_positions = sizes.loc[:prev, need_to_exit].sum(axis=0)
        sizes.loc[trigger_day, need_to_exit] = -1
        logging.debug("exit positions: %s", exit_positions)
        logging.debug("close prices: %s", close_prices.loc[trigger_day, need_to_exit])
    
        need_to_enter = selected > prev_selected
        cash_on_hand = (close_prices.loc[trigger_day, need_to_exit] * exit_positions.values).sum(axis=0)
        top_k_minus_kept = need_to_enter.astype(int).sum()
        if top_k_minus_kept == 0:
            continue
        distr_cash_to_asset = cash_on_hand / top_k_minus_kept
        logging.debug("cash on hand: %s", cash_on_hand)
        logging.debug("top_k_minus_kept: %s", top_k_minus_kept)
        logging.debug("enter price change: %s", price_change.loc[trigger_day, :])
        logging.debug("enter close prices: %s", close_prices.loc[trigger_day, need_to_exit])
        sizes.loc[trigger_day, need_to_enter] = 1.0/top_k_minus_kept
            
        logging.debug("sizes:%s", sizes.loc[trigger_day,:])
        logging.debug("redistributed cash: %s", cash_on_hand)
        logging.debug("exits:%s enters: %s", sizes.columns[need_to_exit], sizes.columns[need_to_enter])

    # close all positions
    sizes.iloc[-1,:] = -1 

    logging.debug("init positions:%s", sizes.iloc[:5,:].to_csv(None, sep='\t', float_format='%.2f'))
    logging.debug("last positions:%s", sizes.iloc[-5:,:].to_csv(None, sep='\t', float_format='%.2f'))
    
    return trigger_close_prices.fillna(0.001), sizes

def simulate(args):
    sp500_tickers = pd.read_csv(f"{DATASET_PATH}/sp_500_basics.csv")
    symbols = [str.lower(s) for s in sp500_tickers['Symbol']]
    #symbols = ['goog', 'msft', 'nvda', 'mmm', 'aa', 'xrx']

    missing_symbols, symbols = StooqData.filter_valid_symbols(symbols)

    logging.info(f"tracking sp500 symbols: {len(symbols)} with {len(missing_symbols)} missing.") 
    stook_data = StooqData.download(symbols, start=args.start_date, end=args.end_date, missing_columns='drop')
    open_prices, high_prices, low_prices, close_prices, volumes = stook_data.get()
    
    rand_gen = np.random.default_rng(args.seed)

    init_cash = 1000
    for r in range(args.repetitions):
        price, size = create_rand_hold_topk_orders(
            rand_gen,
            close_prices,
            top_k=args.top_k,
            approx_triggers_per_year=args.triggers_per_year)
        logging.info(f"triggered {len(price)} times.")

        topk_pf = vbt.Portfolio.from_orders(price,
            size=size,
            init_cash=init_cash,
            size_type=vbt.portfolio.enums.SizeType.Percent,
            direction=vbt.portfolio.enums.Direction.LongOnly,
            cash_sharing=True,
            lock_cash=True,
            fixed_fees=0.01)
   
        if args.out != None:
            topk_pf.save(args.out.format(r))
        
        print(topk_pf.returns_stats(freq='d'))
        print(topk_pf.trades.records_readable.to_csv(None, float_format='%.2f', sep='\t'))

def spy(args):
    spy_data = StooqData.download(['spy','ge'], start=args.start_date, end=args.end_date, missing_columns='drop')
    open_prices, high_prices, low_prices, close_prices, volumes = spy_data.get()
    spy_pf = vbt.Portfolio.from_holding(close_prices.loc[:,['spy',]],
            init_cash=1000,
            cash_sharing=True,
            fixed_fees=0.01)
    print(spy_pf.returns_stats(freq='d'))
    spy_pf.save("00_25_spy.pf")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulate triggering switch at random intervals.")
    parser.add_argument("-k", dest='top_k', default=3, type=int, help='the number of stocks to hold and rebalance. default: 3') 
    parser.add_argument("--start-date", dest='start_date', default="2000-01-01", help='Start date of simulation. default: 2000-01-01') 
    parser.add_argument("--end-date", dest='end_date', default="2025-01-01", help='end date of simulation. default: 2025-01-01') 
    parser.add_argument("--triggers-per-year", dest='triggers_per_year', type=int, default=8, help='number of triggers per year. default: 8') 
    parser.add_argument("--seed", dest='seed', default=42, type=int, help='seed for random. default: 42') 
    parser.add_argument("--debug", dest='level', default=logging.INFO, action='store_const', const=logging.DEBUG, help='writeout debug statements') 
    parser.add_argument("--repetitions", dest='repetitions', default=1, type=int, help='number of repetitions.') 
    parser.add_argument("--out", dest='out', default=None, help='save portfolio(s). If multiple repetitions are specified, then use format string to assign numbers.') 

    args = parser.parse_args()
    logging.basicConfig(level=args.level, format="%(asctime)s:%(process)d:%(levelname)s:%(message)s")

    simulate(args)
    #spy(args)
