import sys
import numpy as np
from keras.models import load_model, Sequential
from keras.layers import deserialize
import matplotlib.pyplot as plt

from data_provider import build_train_data_multiple, get_num_features
from trading import Trader

TEST_CASH = 1000
model_name = sys.argv[1]
test_secs = ["WIKI/F"]

# securities == [stock1_dict, stock2_dict, stock3_dict]
securities = list(
    build_train_data_multiple(test_secs, start_date="2014-01-01", end_date="2017-01-01")
)

# securities = [{'X':[[1, 11, 111, 1111], [2, 22, 222, 2222], ...],
# 				 'price': [1.1, 2.2, ...]},
# 				{'X':[[4, 44, 444, 4444], [5, 55, 555, 5555], ...],
# 				 'price': [4.4, 5.5, ...]}]

_, plots = plt.subplots(len(securities), sharex=True, squeeze=False)
plt.xlabel("Time")
plt.ylabel("Price")

# Y = None
for i, security in enumerate(securities):
    # Y = security['Y']
    testX = security["X"].values
    price = security["price"]
    securities[i] = list(zip(testX, price))
    plots[i][0].plot(range(len(testX)), price, linewidth=3, color="b")

# securities is in the format 	   [[
# 									 (data_AAPL_day1, price_AAPL_day1),
# 									 (data_AAPL_day2, price_AAPL_day2)],
# 									[
# 									 (data_GOOG_day1, price_GOOG_day1),
# 									 (data_GOOG_day2, price_GOOG_day2)
# 								   ]]

days = list(zip(*securities))

# days is in the format 		[ 	[
# 									 (data_AAPL_day1, price_AAPL_day1),
# 									 (data_GOOG_day1, price_GOOG_day1)],
# 									[
# 									 (data_AAPL_day2, price_AAPL_day2),
# 									 (data_GOOG_day2, price_GOOG_day2)],
# 									...more days here
# 								]

all_models = [load_model(model_name)] * len(test_secs)
all_weights = [model.get_weights() for model in all_models]
all_layers = [model.layers for model in all_models]
one_input_length = get_num_features()

for model_idx in range(len(all_models)):
    for i, layer in enumerate(all_layers[model_idx]):
        config = layer.get_config()
        # print(config)
        if "batch_input_shape" in config and one_input_length in config["batch_input_shape"]:
            # first specification of batch_input_shape
            config["batch_input_shape"] = (1, 1, one_input_length)

        if "stateful" in config:
            # if it's a recurrent layer, make it stateful
            config["stateful"] = True

        all_layers[model_idx][i] = deserialize(
            {"class_name": layer.__class__.__name__, "config": config}
        )


all_models = [Sequential(layers) for layers in all_layers]
for model_idx, model in enumerate(all_models):
    model.set_weights(all_weights[model_idx])
    # print(all_models[model_idx].summary())

year_count, day_count = 0, -1
days_in_stk_yr = 252
total_cash = 0
traders = [Trader(init_cash=TEST_CASH, ticker=tick) for tick in test_secs]
prev_trader_values = [0] * len(traders)

# split into trading years
for year in [days[i : i + days_in_stk_yr] for i in range(0, len(days), days_in_stk_yr)]:
    year_count += 1
    print("Now processing year:", year_count)
    for day in year:
        for i, sec in enumerate(day):
            x = np.reshape(sec[0], (1, 1) + np.shape(sec[0]))
            price = round(sec[1], 2)
            prediction = all_models[i].predict(x, batch_size=1)[0][0]
            # prediction = Y[day_count+1]

            trade_made = traders[i].make_trade(prediction, price)

            if trade_made:
                if prediction == 1:  # buy signal
                    plots[i, 0].plot(day_count + 1, price - 0.1, marker="^", color="g", ms=10)
                else:
                    plots[i, 0].plot(day_count + 1, price + 0.1, marker="v", color="r", ms=10)

        day_count += 1

    total_cash = 0
    for i, trader in enumerate(traders):
        val = trader.total_value
        print("Ticker %s made %.2f" % (trader.ticker, val - prev_trader_values[i]))
        prev_trader_values[i] = val
        total_cash += val

    print("Total cash:", round(total_cash, 2))

avg_annual = TEST_CASH + ((total_cash - TEST_CASH) / year_count)
ann_percent = (total_cash - TEST_CASH) / (TEST_CASH * year_count) * 100

print("-------------------------------------------------")
print("|  Initial investment: ${0:.2f}\t\t\t|".format(TEST_CASH))
print("|  Average annual result: ${0:.2f}\t\t|".format(avg_annual))
print("|  Annualized return percent: {0:.2f}%\t\t|".format(ann_percent))
print("-------------------------------------------------")

plt.show()
