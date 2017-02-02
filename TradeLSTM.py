import sys
from Stock_NN_Funcs import build_data
import numpy as np
from keras.models import load_model


TEST_CASH = 10000
model_name = sys.argv[1]
test_secs = ["WIKI/F", "WIKI/GIS", "WIKI/AAPL"]


_build_data = build_data(random_split = False)
_build_data.send(None)
aapl = _build_data.send("WIKI/AAPL")
one_input_length = len(aapl["trX"][0])

def keras_builder(builder):
	# s = yield
	while(1):
		for stock_code in test_secs:
			output = builder.send(stock_code)
			if output is not None:
				x = np.reshape(output["trX"], (1,) + np.shape(output["trX"]))
				y = np.reshape(output["trY"], (1,) + np.shape(output["trY"]))
				# for x, y in zip(x, y)
				yield x, y

builder = keras_builder(_build_data)
builder.send(None)

model = load_model(model_name)

best = 0
for s in test_secs:
	final_cash = 0
	MARGIN_CASH = 0
	chunk = 0
	testX = _build_data.send(s)["X_norm"]
	testX = np.reshape(testX, (1,) + np.shape(testX))
	price = _build_data.send(s)["price"]

	# print(np.shape(price), np.shape(testX))

	output = np.argmax(model.predict(testX, batch_size = len(testX))[0], axis = 1)
	print(output)

	buys, sells = 0, 0
	for prices in [price[i:i+252] for i in range(0, len(price), 252)]:
		CASH = TEST_CASH
		MARGIN_CASH = 10000
		shares = 0
		flag = 0
		short_price = 0
		# print(output)
		# print(price[:30])

		for day_price, bar in zip(prices, output):
			if bar == 0:    #buy
				# # print("buy")
				if flag == 0:       #no position
					buys += 1
					# print("Long")
					shares = CASH / day_price
					CASH -= shares * day_price 
					flag = 1

				# if flag == -1:    #short
				# 	# print("Closing short")
				# 	CASH -= shares * day_price
				# 	shares = 0
				# 	flag = 0

			elif bar == 1:    #sell
				sells += 1
				# print("sell")
				# if flag == 0:       # no position
				# 	# print("Short")
				# 	shares = MARGIN_CASH / day_price
				# 	CASH += shares * day_price
				# 	flag = -1

				if flag == 1:    # long
					# print("Closing long")
					CASH += shares * day_price
					shares = 0
					flag = 0

		if flag == -1:
			CASH -= shares * day_price
		elif flag == 1:
			CASH += shares * day_price
		final_cash += CASH
		if CASH > best:
			global best 
			best = CASH
			# saver.save(sess, "./"+PICKLE_NAME+".ckpt")

		print("Year ", chunk, " returned $", int(CASH), " from an investment of $", TEST_CASH)
		chunk+=1

print("-------------------------------------------------")
print("|  Initial investment:", TEST_CASH, "\t\t\t|")
print("|  Best year result:", int(best), "\t\t\t|")
print("|  Average annual result:", int((final_cash - TEST_CASH)/chunk), "\t\t|")
print("|  Annualized return percent:", int(((final_cash / TEST_CASH) * 100)/chunk)-100, "\t\t|")
print("|  Buys:", buys, " / Sells:", sells, "\t\t\t|")
print("-------------------------------------------------")

