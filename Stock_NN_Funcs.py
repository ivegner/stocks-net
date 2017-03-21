import quandl, sys, os
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import dill as pickle
from sklearn import preprocessing as prep
from sklearn.model_selection import train_test_split
from talib import abstract as ta
from sklearn.externals import joblib
from collections import OrderedDict
import time

np.random.seed(1337) # for reproducibility


quandl.ApiConfig.api_key = "KDH1TFmmmcrjgynvRdWg"

HI_LO_DIFF = 0.03
MIN_MAX_PERIOD = 8

def build_data(raw = False, random_split = True, start_date = None,
			   end_date = None, test_proportion = 0.1):
	# if len(sec) == 1 and os.path.isfile(secs[0]):	#it's a file
	# 	with open(secs[0]) as f:
	# 		secs = ["WIKI/" + line.strip() for line in f]

	# print("SECURITIES: ", s[5:] for s in secs)

	with open("stock_data/invalid_stocks.txt", "r+") as f:
		invalid_stock_codes = [line.strip() for line in f]
	f = open("stock_data/invalid_stocks.txt", "a")

	stock_code = yield

	while True and stock_code is not None:
		valid_stock = False
		while not valid_stock:
			if "." in stock_code:
				stock_code = yield None
				continue
			if stock_code in invalid_stock_codes:
				# print("Skipping security", sec)
				stock_code = yield None
				continue
			valid_stock = True

		sec = stock_code.split("/")[1]	# Just the ticker, not the database code

		pickle_name = sec
		if raw:
			pickle_name += "_raw"
		if not random_split:
			pickle_name += "_notrand"

		if start_date and end_date:
			pickle_name += start_date + "to" + end_date
		elif start_date:
			pickle_name += start_date
		elif end_date:
			pickle_name += "to" + end_date

		if not os.path.isfile("./stock_data/" + pickle_name + "_data.pickle"):
			# print("No pickle found, getting data for", sec)
			try:
				# print("Getting data for", stock_code)
				df = quandl.get(stock_code, start_date = start_date,
								end_date = end_date)

			except quandl.errors.quandl_error.NotFoundError:
				invalid_stock_codes += [stock_code]
				f.write(stock_code + "\n")
				stock_code = yield None
				continue

			if "Adj. Close" in df.columns:
				df = df[["Adj. Open",  "Adj. High",  "Adj. Low",
						 "Adj. Close", "Adj. Volume"]]

				 # Remove the "Adj. " and make lowercase
				df.rename(columns=lambda x: x[5:].lower(), inplace=True)
			elif "Close" in df.columns:
				df = df[["Open",  "High",  "Low",  "Close", "Volume"]]
				# make lowercase
				df.rename(columns=lambda x: x.lower(), inplace=True)

			price = df['close'].values
			minIdxs = argrelextrema(price, np.less)
			maxIdxs = argrelextrema(price, np.greater)


			Y = pd.Series(name="signal", dtype=np.ndarray,
						  index=range(0, len(price)))
			n=0
			for _, idx in np.ndenumerate(minIdxs):
				# if idx < MIN_MAX_PERIOD: continue
				max_price = max(price[idx: idx + MIN_MAX_PERIOD])

				#if the difference between max and min is > X%
				if ((max_price - price[idx]) / price[idx]) > HI_LO_DIFF:
					Y.set_value(idx, np.array([1., 0.], np.float32))
					n+=1

			# print("MINS:", n)
			n=0
			for _, idx in np.ndenumerate(maxIdxs):
				# if idx < MIN_MAX_PERIOD: continue
				min_price = min(price[idx: idx + MIN_MAX_PERIOD])
				#if the difference between max and min is > X%
				if ((price[idx] - min_price)/ min_price) > HI_LO_DIFF:
					Y.set_value(idx, np.array([0., 1.], np.float32))
					n+=1

			# print("MAXS:", n)
			_min_idx, _max_idx = 0, 0
			for i, y in np.ndenumerate(Y.values):
				if np.array_equal(y, [1., 0.]):
					_min_idx = i[0]
				elif np.array_equal(y, [0., 1.]):
					_max_idx = i[0]
				else:
					if _min_idx > _max_idx:
						s =  np.array([1., 0.])
					elif _max_idx > _min_idx:
						s =  np.array([0., 1.])
					else:
						# no action taken
						# only occurs at the beginnings of datasets, afaik
						s = np.array([0., 0.])

					Y.set_value(i, s, np.float32)

			# x = list(zip(price[0:50], Y.values[0:50]))
			# for i in x:
			# 	print("{0:.2f} -- {1}".format(i[0], "sell" if np.array_equal(i[1], [0, 1]) else "buy" if np.array_equal(i[1], [1, 0]) else "nothing"))

			df.reset_index(drop=True, inplace = True)
			if isinstance(price, np.ndarray):
				price = price.tolist()

			''' INDICATORS '''
			# print(len(df), len(Y))
			# print("Building indicators...")
			inputs = df.to_dict(orient="list")
			for col in inputs:
				inputs[col] = np.array(inputs[col])

			c = df["close"]
			for n in range(2, 40):
				inputs["bband_u_"+str(n)], inputs["bband_m_"+str(n)], inputs["bband_l_"+str(n)] = ta.BBANDS(inputs, n)
				inputs["sma_"+str(n)] = ta.SMA(inputs, timeperiod = n)
				inputs["adx_"+str(n)] = ta.ADX(inputs, timeperiod = n)

				fast_ema = c.ewm(span = n, adjust = False).mean()
				slow_ema = c.ewm(span = n*2, adjust = False).mean()
				macd1 = fast_ema - slow_ema
				macd2 = macd1.ewm(span = n*2/3, adjust = False).mean()
				macd3 = macd1 - macd2
				inputs["macd_"+str(n)], inputs["macdsignal_"+str(n)], inputs["macdhist_"+str(n)] = macd1.values, macd2.values, macd3.values
				# macd = [macd1.values, macd2.values, macd3.values]
				# inputs["macd_"+str(n)], inputs["macdsignal_"+str(n)], inputs["macdhist_"+str(n)] = ta.MACD(inputs, n, n*2, n*2/3)

				# for idx, i in enumerate(["macd_"+str(n), "macdsignal_"+str(n), "macdhist_"+str(n)]):
				# 	for day in zip(inputs[i], macd[idx]):
				# 		print("Type: %s N: %d PD: %.3f TA: %.3f, " % (i, n, day[1], day[0]))
				inputs["mfi_"+str(n)] = ta.MFI(inputs, n)
				inputs["ult_"+str(n)] = ta.ULTOSC(inputs, n, n*2, n*4)
				inputs["willr_"+str(n)] = ta.WILLR(inputs, n)
				inputs["slowk"], inputs["slowd"] = ta.STOCH(inputs)
				inputs["mom_"+str(n)] = ta.MOM(inputs, n)

			inputs["volume"] = list(map(lambda x: x/10000, inputs["volume"]))
			# print(inputs["macd_2"], inputs["macdsignal_2"], inputs["macdhist_2"])

			df = pd.DataFrame().from_dict(inputs)
			broken = False

			for idx, val in reversed(list(df.isnull().any(axis=1).iteritems())):
				if val == True:
					# print(actual_idx, val)
					df.drop(idx, inplace = True)
					Y.drop(idx, inplace = True)
					try:
						# price[actual_idx] = None
						price.pop(idx)
					except IndexError:	#drop the security
						# print("Error, dropping security", sec)
						broken = True
						break

			# print(list(df.isnull().any(axis=1).iteritems()))
			# print("PRICES", price)

			# print(len(price), len(df.values))

			# for i, p in reversed(list(enumerate(price))):
			# 	actual_idx = len(price) - 1 - i
			# 	if p is None:
			# 		print(actual_idx)
			# 		price.pop(actual_idx)

			''' BUILD NEURAL NET INPUTS '''
			if not broken:
				Y = np.vstack(Y.values)
				# print(df["adx_10"])
				X = df.values
				# print(X[0:2])

				if not raw:
					rand = "_notrand" if not random_split else ""

					if not os.path.isfile("./stock_data/" + sec + rand + ".scaler"):
						scaler = prep.StandardScaler().fit(X)
						X_norm = scaler.transform(X)
						joblib.dump(scaler, "./stock_data/" + sec + rand + ".scaler")
					else:
						scaler = joblib.load("./stock_data/" + sec + rand + ".scaler")
						X_norm = scaler.transform(X)

				else:
					X_norm = X

				if random_split:
					trX, testX, trY, testY = train_test_split(X_norm, Y, test_size = test_proportion, random_state=0)

				else: 		# just clips the test data off the end
					l = len(X_norm)
					trX, testX = X_norm[:int(-test_proportion*l)], X_norm[int(-test_proportion*l):]
					trY, testY = Y[:int(-test_proportion*l)], Y[int(-test_proportion*l):]

				# print("Pickling...")

				output = {"X_norm": X_norm, "Y": Y, "trX": trX, "trY": trY, "testX": testX, "testY": testY, "price": price}
				pickle.dump(output, open("./stock_data/" + pickle_name + "_data.pickle", "wb"))
				stock_code = yield output

			else:
				invalid_stock_codes += [stock_code]
				f.write(stock_code + "\n")
				stock_code = yield None


		else:
			# print("Pickle found, loading...")

			_data = pickle.load(open("./stock_data/" + pickle_name + "_data.pickle", "rb"))
			trX, trY, testX, testY, price, X_norm, Y = _data["trX"], _data["trY"], _data["testX"], _data["testY"], _data["price"], _data["X_norm"], _data["Y"]
			stock_code = yield {"X_norm": X_norm, "Y": Y, "trX": trX, "trY": trY, "testX": testX, "testY": testY, "price": price}


def build_realtime_data(stock_codes, start_date = None, end_date = None, backload = 700):

	with open("stock_data/invalid_stocks.txt", "r+") as f:
		invalid_stock_codes = [line.strip() for line in f]
	f = open("stock_data/invalid_stocks.txt", "a")

	for code in stock_codes:
		if code in invalid_stock_codes:
			print("Invalid stock")
			sys.exit(1)

	testing_mode = True if start_date else False

	builder = _build_indicators(len(stock_codes))
	builder.send(None)

	# Setup LSTM memory	(:-1 because the last entry would be the current day)
	# raw_price_data_backload is a list of DataFrames
	raw_price_data_backload = [quandl.get(stock, end_date = start_date, limit = backload)[:-1] for stock in stock_codes]

	historical_backload = OrderedDict()

	for day_idx in raw_price_data_backload[0].index.values:
		day = []
		for sec_idx in range(len(stock_codes)):
			day += [raw_price_data_backload[sec_idx].loc[day_idx]]

		day_data = builder.send(day)
		incomplete_data = False
		for datum in day_data[0]["data"]:
			if np.isnan(datum):
				incomplete_data = True
				break

		if not incomplete_data:		# only add those that don't have NaN's
			historical_backload[day_idx] = day_data

	''' historical_backload == [
								[date, {"data": stock1_day1_data, "price": stock1_day1_price}, {...stock2 day 1...}],
								[date, {"data": stock1_day2_data, "price": stock1_day2_price}, {...stock2 day 2...}],
								...
							   ]
	'''


	# for i in range(len(stock_codes)):
	# 	print(backloaded_data[i])
	# 	assert len(backloaded_data[i]) == len(raw_price_data_backload[i])

	# scalers = {stock_code: prep.StandardScaler().fit(backloaded_data[0]) for stock_code in stock_codes}

	scalers = [None for stock_code in stock_codes]

	for day_idx, day in historical_backload.items():
		for i, stock_code in enumerate(stock_codes):

			sec = stock_code.split("/")[1]	# Just the ticker, not the database code

			if not scalers[i]:
				if not os.path.isfile("./stock_data/" + sec + "_notrand.scaler"):
					print("No scaler for", sec)
					sys.exit(1)
				else:
					scaler = joblib.load("./stock_data/" + sec + "_notrand.scaler")
					scalers[i] = scaler
			else:
				scaler = scalers[i]

			# print("Before", historical_backload[day_idx][i]["data"][0])
			# print(np.isnan(historical_backload[day_idx][i]["data"]).any())
			historical_backload[day_idx][i]["data"] = scaler.transform([historical_backload[day_idx][i]["data"]])[0]
			# print("After", historical_backload[day_idx][i]["data"][0])


	yield historical_backload	# this is yielded when None is sent to the generator, i.e. on the first run

	''' historical_backload == [
								[date, [{"data": stock1_day1_data, "price": stock1_day1_price}, {...stock2 day 1...}]],
								[date, [{"data": stock1_day2_data, "price": stock1_day2_price}, {...stock2 day 2...}]],
								...
							   ]
	'''


	if testing_mode:	# pre-load a bunch of historical data to simulate real-time trades
		raw_historical_prices = [quandl.get(stock, start_date = start_date, end_date = end_date) for stock in stock_codes]
		historical_price_data = OrderedDict()

		for day_idx in raw_historical_prices[0].index.values:
			day = []
			for sec_idx in range(len(stock_codes)):
				day += [raw_historical_prices[sec_idx].loc[day_idx]]
			historical_price_data[day_idx] = day

		historical_price_data = list(historical_price_data.items())

	day_counter = 0
	# Hand out data one-by-one

	print("LENGTH OF HISTORICAL", len(historical_price_data))
	while True:
		if not testing_mode:	# if no pre-built prices, i.e. actually realtime
			raw_price_data = [quandl.get(stock, limit = 1) for stock in stock_codes]	# just the last entry
			day_data = []
			for sec_idx in range(len(stock_codes)):
				day_data += [raw_price_data[sec_idx]]

			date = raw_price_data[0].index.values[0]
		else:
			try:
				day_data = historical_price_data[day_counter][1]
			except IndexError:
				break
			date = historical_price_data[day_counter][0]

		# print("Trading day #{}".format(day_counter), "date:", date)

		day_data = builder.send(day_data)


		for i in range(len(stock_codes)):
			# print(np.isnan(day_data[i]["data"]).any())
			day_data[i]["data"] = scalers[i].transform([day_data[i]["data"]])[0]

		day_counter += 1

		yield (date, day_data)

def _build_indicators(num_secs):	# accepts a list of one-day Series

	sec_idx_range = range(num_secs)
	sliding_window = []	# list of pd.DataFrames

	data = yield

	for datum in data:
		sliding_window += [_rename_columns(datum)]

	current_day = 0
	while True:
		from talib import abstract as ta
		passes_validity_check, num_validation_iterations = False, 0
		# time.sleep(1)
		while not passes_validity_check:
			for i in sec_idx_range:	# for each security
				# print("Current day:", current_day)
				if current_day != 0:
					if current_day > 170 and num_validation_iterations == 0:
						sliding_window[i] = sliding_window[i].iloc[1:]	# pop the first

					for datum in data:
						if num_validation_iterations == 0:
							sliding_window[i] = sliding_window[i].append(_rename_columns(datum))

				data_with_ind = []

				series = sliding_window[i]
				series = series.reset_index(drop=True)

				inputs = series.to_dict(orient="list")
				for col in inputs:
					inputs[col] = np.array(inputs[col])

				c = series["close"]
				for n in range(2, 40):
					inputs["bband_u_"+str(n)], inputs["bband_m_"+str(n)], inputs["bband_l_"+str(n)] = ta.BBANDS(inputs, n)
					inputs["sma_"+str(n)] = ta.SMA(inputs, timeperiod = n)
					inputs["adx_"+str(n)] = ta.ADX(inputs, timeperiod = n)

					fast_ema = c.ewm(span = n).mean()
					slow_ema = c.ewm(span = n*2).mean()
					macd1 = fast_ema - slow_ema
					macd2 = macd1.ewm(span = n*2/3).mean()
					macd3 = macd1 - macd2
					inputs["macd_"+str(n)], inputs["macdsignal_"+str(n)], inputs["macdhist_"+str(n)] = macd1.values, macd2.values, macd3.values

					inputs["mfi_"+str(n)] = ta.MFI(inputs, n)
					inputs["ult_"+str(n)] = ta.ULTOSC(inputs, n, n*2, n*4)
					inputs["willr_"+str(n)] = ta.WILLR(inputs, n)
					inputs["slowk"], inputs["slowd"] = ta.STOCH(inputs)
					inputs["mom_"+str(n)] = ta.MOM(inputs, n)
					inputs["mom_"+str(n)] = ta.MOM(inputs, n)

				inputs["volume"] = list(map(lambda x: x/10000, inputs["volume"]))

				# print(len(inputs), len(inputs["close"]), len(inputs["macd_2"]))
				series = pd.DataFrame().from_dict(inputs)

				price = series["close"].iloc[-1]
				if isinstance(price, np.ndarray):
					price = price.tolist()


				# for idx, val in series.isnull().any(axis=1).iteritems():
				# 	if val == True:
						# series.drop(idx, inplace = True)
						# try:
						# 	price[idx] = None
						# except IndexError:	#drop the security
						# 	print("Error, failed to drop price on index", idx)
						# 	sys.exit(1)
						# # print("Dropped index:", idx)

				# for i, p in reversed(list(enumerate(price))):
				# 	actual_idx = len(price) - 1 - i
				# 	if p == None:
				# 		price.pop(actual_idx)

				# print(series["adx_10"])
				X = series.iloc[-1].values

				if not np.isnan(X).any() or current_day < 170:
					passes_validity_check = True

				else:
					num_validation_iterations += 1
					print("Reevaluating, iteration", num_validation_iterations)

				# if np.isnan(X).any() and current_day > 170:
				# 	# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
				# 	# 	print(series)
				# 	print(sliding_window[0])
				# 	break

				# print("ADX_10:\n", series["adx_10"].tail(3))

				# if current_day == 900:
				# 	print(series)
				# 	print(X)

				data_with_ind += [{"data": X, "price": round(price, 2)}]

		data = yield data_with_ind
		current_day += 1


def _rename_columns(df):
	if isinstance(df, pd.core.series.Series):
		df = pd.DataFrame([df], columns = df.index.values)

	if "Adj. Close" in df.columns:
		df = df[["Adj. Open",  "Adj. High",  "Adj. Low",  "Adj. Close", "Adj. Volume"]]
		df.rename(columns=lambda x: x[5:].lower(), inplace=True)    # Remove the "Adj. " and make lowercase
	elif "Close" in df.columns:
		df = df[["Open",  "High",  "Low",  "Close", "Volume"]]
		df.rename(columns=lambda x: x.lower(), inplace=True)    # make lowercase

	return df
