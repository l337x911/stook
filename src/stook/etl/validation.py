import vectorbt as vbt
import numpy as np

from stook.etl.dataset import StooqData


def consistency_check(src, target, symbols, start, end, src_to_value, target_to_value, src_kwargs={}, target_kwargs={}):
    """Checks if src and target data downloaded values are similar by reporting their MSE.
    :src: vbt data object for comparison
    :target: vbt data object for comparison
    :start: start date for downloading pricing.
    :end: end date for downloading pricing.
    :src_to_value: mapping of data get to a price for comparison.
    :target_to_value: mapping of data get to a price for comparison.
    """
    src_data = src.download(symbols, start=start, end=end, **src_kwargs)
    src_values = src_to_value(src_data)
    target_data = target.download(symbols, start=start, end=end, **target_kwargs)
    target_values = target_to_value(target_data)
    #print(src_values)
    #print(target_values)
    mean_square_error = np.mean((src_values.values - target_values.values)**2)
    print(f"MSE:{mean_square_error}")

#print(StooqData.download(['GOOG', 'MSFT'], missing_index='drop', start="2024-01-01", end="2024-06-01").get())

#print(vbt.YFData.download(['GOOG', 'MSFT'], missing_index='drop', start="2024-01-01", end="2024-06-01", interval='1d').get())


consistency_check(StooqData,
    vbt.YFData, # yf reports pricing adjusted for splits, dividends, and capital gains.
    ["GOOG","MSFT"],
    "2024-01-01",
    "2024-06-01",
    lambda x: x.get()[0],
    lambda x: x.get("Open"),
    src_kwargs= {'missing_index':'drop'},
    target_kwargs= {'missing_index':'drop', 'interval':'1d'}
)
