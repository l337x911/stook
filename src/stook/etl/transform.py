import pandas as pd


def price_change(prices, lag=1):
    """Transforms a dataframe of prices on n days x m assets to a price change on a lag of days."""
    price_change = prices.iloc[lag:,:].copy()
    return price_change / prices.iloc[:-lag,:]
