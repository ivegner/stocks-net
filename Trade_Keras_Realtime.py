import sys
from Stock_NN_Funcs import build_realtime_data
import numpy as np
from keras.models import load_model, Sequential
from keras.layers import Recurrent
from keras.utils.layer_utils import layer_from_config
import math
import matplotlib.pyplot as plt

np.random.seed(1337) # for reproducibility

TEST_CASH = 100

cash = TEST_CASH
model_name = sys.argv[1]
test_secs = ["WIKI/AAPL", "WIKI/XOM", "WIKI/F"]
trading_days_in_year = 252

builder = build_realtime_data(test_secs, start_date = "2014-01-01", end_date = "2016-01-01", backload = 200)

backload_data = builder.send(None)	# OrderedDict [(date, [{data: data1, price:price1}, {data: data2, price:price2}]), (date2, ...)]
backload_data = list(backload_data.items())
one_input_length = len(backload_data[0][1][0]["data"])	# dat indexing tho
print(one_input_length)

_, plots = plt.subplots(len(test_secs), sharex=True, squeeze = False)
plt.xlabel("Time")
plt.ylabel("Price")

all_models = [load_model(model_name)]*len(test_secs)
all_weights = [model.get_weights() for model in all_models]
all_layers = [model.layers for model in all_models]

for model_idx in range(len(all_models)):
	for i, layer in enumerate(all_layers[model_idx]):
		config = layer.get_config()
		if "batch_input_shape" in config and one_input_length in config["batch_input_shape"]:	# first specification of batch_input_shape
			config["batch_input_shape"] = (1, 1, one_input_length)

		if isinstance(layer, Recurrent): # if it's a recurrent layer, make it stateful
			config["stateful"] = True

		all_layers[model_idx][i] = layer_from_config({"class_name": type(layer), "config": config})

all_models = [Sequential(layers) for layers in all_layers]
for model_idx in range(len(all_models)):
	all_models[model_idx].set_weights(all_weights[model_idx])


# def execute_trade(type, price, cash):
# 	if type == 0:	# buy


buys, sells = 0, 0

shares = None
bought_ticker = None
bought_flag = 0	# 1 for active trade, 0 for no active trades

continue_trading = True

prices = [[-999] * len(test_secs)]

# print("Pretraining...")
# for _, data in backload_data:
# 	# data is an array of dicts, with each containing "data" and "price" for one security
# 	for i, security in enumerate(data):
# 		security = np.reshape(security["data"], (1, 1,) + np.shape(security["data"]))
# 		all_models[i].predict(security, batch_size = 1)

day_count = 0
while(continue_trading):
	try:
		day = next(builder)
		# day = backload_data[0:4][day_count]
		# day is a tuple (date, array), where array is an array of {"data":data, "price":price} for each security
	except StopIteration:
		break

	date, data = day

	for i, sec in enumerate(data):
		# sec is {"data":data, "price":price}

		x = np.reshape(sec["data"], (1, 1,) + np.shape(sec["data"]))
		price = sec["price"]
		prediction = np.argmax(all_models[i].predict(x, batch_size = 1)[0][0])
		print("Prediction:", prediction, "price", price, "date", date)

		if prices[i][0] == -999:
			prices[i][0] = price
		else:
			prices[i] += [price]

		if prediction == 0:	# buy
			if bought_flag == 0:       #no position
				buys += 1
				# print("Long on ticker", bought_ticker)
				shares = math.floor(cash / price)
				cash -= shares * price
				cash = round(cash, 2)
				print("BUY:\tShares:", shares, "\tprice", price, "\tof ticker", i, "\tRemaining cash:", cash)
				bought_flag = 1
				bought_ticker = i 
				# plots[i, 0].axvline(x=day_count, color="g")
				plots[i, 0].plot(day_count, price - 0.2, marker = "^", color = "g", ms = 5)

		elif prediction == 1:	# sell
			if bought_flag == 1:       # long position
				if i == bought_ticker:
					# print("Closed long on ticker", bought_ticker)
					sells += 1
					profit = round(shares * price, 2)
					print("SELL:\tShares:", shares, "\tprice", price, "\tof ticker", bought_ticker, "\tResult from trade:", profit)
					cash += profit
					cash = round(cash, 2)
					print("Current cash:", cash)
					shares = 0
					bought_flag = 0
					bought_ticker = None
					# plots[i, 0].axvline(x=day_count, color="r")
					plots[i, 0].plot(day_count, price + 0.2, marker = "v", color = "r", ms = 5)

		# print(sec["data"][0:4])

	day_count += 1

for i in range(len(test_secs)):
	plots[i, 0].plot(range(day_count), prices[i], linewidth = 3)

print("-------------------------------------------------")
print("|  Initial investment:", TEST_CASH, "\t\t\t|")
# print("|  Best year result:", int(best), "\t\t\t|")
print("|  Average annual result:", round((cash - TEST_CASH)/(day_count/trading_days_in_year)), "\t\t|")
print("|  Annualized return percent:", round(((cash / TEST_CASH) * 100)/(day_count/trading_days_in_year))-100, "\t\t|")
print("|  Buys:", buys, " / Sells:", sells, "\t\t\t|")
print("-------------------------------------------------")

plt.show()







