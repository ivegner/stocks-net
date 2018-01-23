# StocksNet
*A neural network-based stock trading algorithm*

## Usage
**To be able to get data from Quandl, obtain a Quandl API key and put it in `quandl_key.txt`**

To train:
    For a full list of CLI args, call `python train_lstm.py --help`

    Example: `python3 train_lstm.py -e 50 -l 4 -n 1000 WIKI/MMM WIKI/ABT WIKI/ABBV` trains a neural net with 4 recurrent layers with 1000 cells each for 50 epochs, using data from the securities MMM, ABT, and ABBV

To test:
    `python trade_lstm.py <filename of saved model>`. The stock being used to test is currently hard-coded

## What it does
* Uses technical indicators and price data to predict the direction of stock movement and major price direction changes (peaks/troughs).
* Based on this output, makes decisions on whether to buy/sell (i.e. if the predicted direction of prices is up, go long). Tries to buy at the troughs and sell at the peaks.

## How training works
1. Training algorithm (trainer) is called, parameters of the neural net are specified. (**filename here**)
1. Trainer builds training data with `data_provider.build_train_data`, which:
    1. Pulls (daily) price data from Quandl.
    1. Processes the data, builds indicators, aggregates into Pandas dataframes. *(Processed data is cached by default so as to save resources, be careful with big datasets and little disk space.)*
    1. Creates the output labels (i.e. the direction of stock movement, 1 for up/long, 0 for down/short)
    1. Splits data into training and testing, optionally scaling/shuffling/etc.
    1. Returns Pandas Dataframes of `train_x`, `test_x` and Series of ` train_y` and `test_y`.
1. Trains neural net as specified on training data, tests on the testing data. Uses Keras to build the neural nets.
1. Optionally, saves intermediate training results, outputs accuracy, and/or simulates "live trading" with another stock ticker (preferably one it hasn't seen before).

## How "live trading" testing works
1. Takes parameters including the stock ticker to use, how much cash to start with, the period to test on.
1. Builds data as described above. Does not build labels, as this will not be used for training, but rather for simulating "real-life" trading, in which the correct action is obviously not provided.
1. Steps through the data day-by-day, making long-short decisions based on the output of the network being tested.
1. Outputs yearly results, annualized return percentage and (optionally) individual trade details.