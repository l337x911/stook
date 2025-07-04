import unittest
import vectorbt as vbt
from stook.etl.dataset import StooqData
import pandas as pd
import numpy as np

MAX_ERROR = 0.0001

class PortfolioTests(unittest.TestCase):
    """ Holds test to validate portfolio metrics. """

    def test_annualized_returns(self):
        """Tests annualized return for multi-asset portfolio."""
        symbols = ['SPY', 'GOOG']
        _, _, _, close_price, _ = StooqData.download(symbols, missing_index='drop', start="2024-01-01", end="2024-06-01")\
                .get()
        n = close_price.shape[0]

        pf = vbt.Portfolio.from_holding(close_price, init_cash=50, group_by=True)
        total_return = pf.total_return()
        annualized_return = pf.annualized_return(freq='d', year_freq='252d')
        # annualized return is overestimated as the span of trading days should be included (252 in year rather than 365).
        true_annualized_return = ((1 + total_return) ** (252 / n) - 1) * 100
        self.assertTrue((annualized_return - true_annualized_return) < MAX_ERROR)


    def test_yfdata_annualized_returns(self):
        """Tests annualized return for multi-asset portfolio."""
        
        close_price = vbt.YFData.download(['GOOG', 'MSFT'], missing_index='drop', start="2024-01-01", end="2024-06-01", interval='1d').get("Close")
        pf = vbt.Portfolio.from_holding(close_price, init_cash=50)
        total_return = pf.total_return()
        annualized_return = pf.annualized_return(freq='d', year_freq='252d')
        # annualized return is overestimated as the span of trading days should be included (252 in year rather than 365). Fixed by specifying year_freq.
        n = close_price.shape[0]
        true_annualized_return = ((1 + total_return) ** (252 / n) - 1)
        true_annualized_return.name = 'annualized_return'
        #print(f"true for span:\n{true_annualized_return}\nreported:\n{annualized_return}")
        self.assertTrue(np.mean((true_annualized_return - annualized_return)**2) < MAX_ERROR)    
        # averaged annualized returns is not the same as annualized returns.
        # fixed by group_by of assets as single
        pf = vbt.Portfolio.from_holding(close_price, init_cash=50, group_by=True)
        total_return = pf.total_return()

        mean_stats = pf.returns_stats(freq='d', year_freq='252d')
        true_annualized_return = ((1 + (mean_stats['Total Return [%]']/100)) ** (252 / n) - 1)
        #print(f"{total_return} reported:{mean_stats['Total Return [%]']}")
        #print(f"true avg:{true_annualized_return} reported:{mean_stats['Annualized Return [%]']}")
        self.assertTrue((mean_stats['Total Return [%]'] - 100*total_return) < MAX_ERROR)
        self.assertTrue((mean_stats['Annualized Return [%]'] - 100*true_annualized_return) < MAX_ERROR)
         

if __name__ == '__main__':
    unittest.main()
