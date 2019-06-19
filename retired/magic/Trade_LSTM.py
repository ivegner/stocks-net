import sys
from Stock_NN_Funcs import build_data
import numpy as np
from keras.models import load_model, Sequential
from keras.layers import Recurrent
from keras.utils.layer_utils import layer_from_config
import math
import matplotlib.pyplot as plt

TEST_CASH = 100
model_name = sys.argv[1]
test_secs = ["WIKI/CZR"]
trading_days_in_year = 252


# _build_data = build_data(random_split = False, start_date = "1999-01-01", end_date = "2017-01-01")
_build_data = build_data(random_split=False, start_date="2014-01-01", end_date="2016-01-01")
_build_data.send(None)
# aapl = _build_data.send("WIKI/AAPL")
one_input_length = 463  # len(aapl["trX"][0])

# def keras_builder(builder):
# 	# s = yield
# 	while(1):
# 		for stock_code in test_secs:
# 			output = builder.send(stock_code)
# 			if output is not None:
# 				x = np.reshape(output["X_norm"], (1,) + np.shape(output["X_norm"]))
# 				y = np.reshape(output["trY"], (1,) + np.shape(output["trY"]))
# 				# for x, y in zip(x, y)
# 				yield x, y

best = 0
securities = list(
    map(lambda s: _build_data.send(s), test_secs)
)  # securities == [stock1_dict, stock2_dict, stock3_dict]

# securities = [{"X_norm":[[1, 11, 111, 1111], [2, 22, 222, 2222], [3, 33, 333, 3333]], "price": [1.1, 2.2, 3.3]}, {"X_norm":[[4, 44, 444, 4444], [5, 55, 555, 5555], [6, 66, 666, 6666]], "price": [4.4, 5.5, 6.6]}]

_, plots = plt.subplots(len(securities), sharex=True, squeeze=False)
plt.xlabel("Time")
plt.ylabel("Price")

for i in range(len(securities)):
    testX = securities[i]["X_norm"].tolist()
    price = securities[i]["price"]
    # testX = np.reshape(testX, (1,) + np.shape(testX)).tolist()
    securities[i] = list(zip(testX, price))
    plots[i][0].plot(range(len(testX)), price, linewidth=3)

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

for model_idx in range(len(all_models)):
    for i, layer in enumerate(all_layers[model_idx]):
        config = layer.get_config()
        if (
            "batch_input_shape" in config and one_input_length in config["batch_input_shape"]
        ):  # first specification of batch_input_shape
            config["batch_input_shape"] = (1, 1, one_input_length)

        if isinstance(layer, Recurrent):  # if it's a recurrent layer, make it stateful
            config["stateful"] = True

        all_layers[model_idx][i] = layer_from_config({"class_name": type(layer), "config": config})

all_models = [Sequential(layers) for layers in all_layers]
for model_idx in range(len(all_models)):
    all_models[model_idx].set_weights(all_weights[model_idx])
    # print(all_models[model_idx].summary())

total_period = len(days)

buys, sells = 0, 0

shares = None
bought_ticker = None
bought_flag = 0  # 1 for active trade, 0 for no active trades

year_count, day_count, total_cash, last_price = 0, -1, 0, 0

# for i, sec in enumerate(securities):	#plot the daily prices
# 	print([day[1] for day in sec])
# 	plots[i, 0].plot(x =range(len(sec)), y = [day[1] for day in sec], color = "b")

for year in [
    days[i : i + trading_days_in_year] for i in range(0, total_period, trading_days_in_year)
]:  # split into trading years
    cash = TEST_CASH
    year_count += 1
    print("Now processing year:", year_count)
    for day in year:
        for i, sec in enumerate(day):
            # print(i, sec)
            x = np.reshape(sec[0], (1, 1) + np.shape(sec[0]))
            # print(np.shape(x))
            # print(sec)
            price = round(sec[1], 2)
            # print(x)
            prediction = np.argmax(all_models[i].predict(x, batch_size=1)[0][0])

            if prediction == 0:  # buy
                if bought_flag == 0:  # no position
                    buys += 1
                    # print("Long on ticker", bought_ticker)
                    shares = math.floor(cash / price)
                    cash -= shares * price
                    cash = round(cash, 2)
                    print(
                        "BUY:\tShares:",
                        shares,
                        "\tprice",
                        price,
                        "\tof ticker",
                        i,
                        "\tRemaining cash:",
                        cash,
                    )
                    bought_flag = 1
                    bought_ticker = i
                    # plots[i, 0].axvline(x=day_count, color="g")
                    plots[i, 0].plot(day_count, price - 0.2, marker="^", color="g", ms=5)

            elif prediction == 1:  # sell
                if bought_flag == 1:  # long position
                    if i == bought_ticker:
                        # print("Closed long on ticker", bought_ticker)
                        sells += 1
                        profit = round(shares * price, 2)
                        print(
                            "SELL:\tShares:",
                            shares,
                            "\tprice",
                            price,
                            "\tof ticker",
                            bought_ticker,
                            "\tResult from trade:",
                            profit,
                        )
                        cash += profit
                        cash = round(cash, 2)
                        print("Current cash:", cash)
                        shares = 0
                        bought_flag = 0
                        bought_ticker = None
                        # plots[i, 0].axvline(x=day_count, color="r")
                        plots[i, 0].plot(day_count, price + 0.2, marker="v", color="r", ms=5)

                        # plots[i, 0].plot(day_count, price, color="y")
        day_count += 1

    if bought_flag == 1:
        last_price = round(year[-1][bought_ticker][1], 2)
        cash += shares * last_price
        shares = 0
        bought_flag = 0
        sells += 1

    total_cash += cash
    print("Return:", round(cash, 2))
    print("Total cash:", round(total_cash, 2))

plt.show()

print("-------------------------------------------------")
print("|  Initial investment:", TEST_CASH, "\t\t\t|")
# print("|  Best year result:", int(best), "\t\t\t|")
print("|  Average annual result:", int(total_cash / year_count), "\t\t|")
print(
    "|  Annualized return percent:",
    int(((total_cash / TEST_CASH) * 100) / year_count) - 100,
    "\t\t|",
)
print("|  Buys:", buys, " / Sells:", sells, "\t\t\t|")
print("-------------------------------------------------")

# import sys
# from Stock_NN_Funcs import build_data
# import numpy as np
# from keras.models import load_model


# TEST_CASH = 100
# model_name = sys.argv[1]
# test_secs = ["WIKI/XOM"]


# _build_data = build_data(random_split = False, start_date = "1999-01-01", end_date = "2017-01-01")
# _build_data.send(None)
# aapl = _build_data.send("WIKI/AAPL")
# one_input_length = len(aapl["trX"][0])

# def keras_builder(builder):
# 	# s = yield
# 	while(1):
# 		for stock_code in test_secs:
# 			output = builder.send(stock_code)
# 			if output is not None:
# 				x = np.reshape(output["trX"], (1,) + np.shape(output["trX"]))
# 				y = np.reshape(output["trY"], (1,) + np.shape(output["trY"]))
# 				# for x, y in zip(x, y)
# 				yield x, y

# builder = keras_builder(_build_data)
# builder.send(None)

# model = load_model(model_name)

# best = 0
# for s in test_secs:
# 	final_cash = 0
# 	MARGIN_CASH = 0
# 	chunk = 0
# 	testX = _build_data.send(s)["X_norm"]
# 	testX = np.reshape(testX, (1,) + np.shape(testX))
# 	price = _build_data.send(s)["price"]

# 	# print(np.shape(price), np.shape(testX))

# 	output = np.argmax(model.predict(testX, batch_size = len(testX))[0], axis = 1)
# 	# print(o for o in output)

# 	buys, sells = 0, 0
# 	for prices in [price[i:i+trading_days_in_year] for i in range(0, len(price), trading_days_in_year)][0:1]:
# 		CASH = TEST_CASH
# 		MARGIN_CASH = 10000
# 		shares = 0
# 		flag = 0
# 		short_price = 0
# 		# print(output)
# 		# print(price[:30])

# 		for day_price, bar in zip(prices, output):
# 			if bar == 0:    #buy
# 				if flag == 0:       #no position
# 					print("buy: price", day_price)

# 					buys += 1
# 					# print("Long")
# 					shares = CASH / day_price
# 					CASH -= shares * day_price
# 					flag = 1

# 				# if flag == -1:    #short
# 				# 	# print("Closing short")
# 				# 	CASH -= shares * day_price
# 				# 	shares = 0
# 				# 	flag = 0

# 			elif bar == 1:    #sell
# 				# print("sell")
# 				# if flag == 0:       # no position
# 				# 	# print("Short")
# 				# 	shares = MARGIN_CASH / day_price
# 				# 	CASH += shares * day_price
# 				# 	flag = -1

# 				if flag == 1:    # long
# 					print("sell: price", day_price)
# 					# print("Closing long")
# 					sells += 1
# 					CASH += shares * day_price
# 					shares = 0
# 					flag = 0

# 		if flag == -1:
# 			CASH -= shares * day_price
# 		elif flag == 1:
# 			CASH += shares * day_price
# 		final_cash += CASH
# 		if CASH > best:
# 			global best
# 			best = CASH
# 			# saver.save(sess, "./"+PICKLE_NAME+".ckpt")

# 		print("Year ", chunk, " returned $", int(CASH), " from an investment of $", TEST_CASH)
# 		chunk+=1

# print("-------------------------------------------------")
# print("|  Initial investment:", TEST_CASH, "\t\t\t|")
# print("|  Best year result:", int(best), "\t\t\t|")
# print("|  Average annual result:", int((final_cash - TEST_CASH)/chunk), "\t\t|")
# print("|  Annualized return percent:", int(((final_cash / TEST_CASH) * 100)/chunk)-100, "\t\t|")
# print("|  Buys:", buys, " / Sells:", sells, "\t\t\t|")
# print("-------------------------------------------------")
