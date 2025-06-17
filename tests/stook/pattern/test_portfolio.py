import unittest
import vectorbt as vbt
from stook.etl.dataset import StooqData

class PortfolioTests(unittest.TestCase):
    """ Holds test to validate portfolio metrics. """

    def test_annualized_returns(self):
        """Tests annualized return for multi-asset portfolio."""
        symbols = ['spy', 'goog']
        _, _, _, close_price, _ = StooqData.download(symbols, missing_index='drop', start="2024-01-01", end="2024-06-01")\
                .get()

        pf = vbt.Portfolio.from_holding(close_price, init_cash=50)
        total_return = pf.total_return()
        annualized_return = pf.annualized_return(freq='d')
        # annualized return is overestimated as the span of trading days should be included (252 in year rather than 365).
        true_annualized_return = ((1 + (0.01 * total_return)) ** (252 / close_price.shape[0]) - 1) * 100
        self.assertTrue((annualized_return > true_annualized_return).all())

        self.assertTrue((total_return < annualized_return).all())

        # averaged annualized returns is not the same as annualized returns.
        mean_stats = pf.returns_stats(freq='d')
        true_annualized_return = ((1 + (0.01 * mean_stats['Total Return [%]'])) ** (365 / close_price.shape[0]) - 1) * 100
        self.assertTrue(mean_stats['Annualized Return [%]'] > true_annualized_return)
        

if __name__ == '__main__':
    unittest.main()
