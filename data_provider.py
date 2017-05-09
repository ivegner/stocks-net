import os
import copy
import time
import numpy as np
import pandas as pd
import dill as pickle
from scipy.signal import argrelextrema
from sklearn import preprocessing as prep
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import quandl
from talib import set_unstable_period
from talib import stream
from talib import abstract as ta

quandl.ApiConfig.api_key = "KDH1TFmmmcrjgynvRdWg"
np.random.seed(1337) # for reproducibility

HI_LO_DIFF = 0.03
MIN_MAX_PERIOD = 8
# set_unstable_period("EMA", 100)

class DataBuilder():
	""" 
	Builds stock data
	Init: 
		realtime: Whether to give all the data back at once (F) or one-by-one (T)
		raw: Whether to scale the data
		random_split: Whether to split train and test randomly
		test: Whether to make test data
		test_split: how much of the data to leave to testing
	"""
	def __init__(self, realtime = False, raw = False, 
				 random_split = True, test = False, 
				 test_split = 0.1, mode="train"):
		self.realtime = realtime
		self.raw = raw
		self.random_split = random_split
		self.test = test
		self.test_split = test_split
		self.mode = mode

	"""
	Computes and returns indicators for given stock tickers
	Args:
		stocks: array of Quandl-compatible stock codes
		start_date, end_date: "YYYY-MM-DD"
		mode ("train" | "test"): whether to build Y-targets for supervised training

	Returns:
		if self.realtime:
			Array of 1-day results for each stock 
			[[STOCK1_DAY1_DATA], [STOCK2_DAY1_DATA]...]
		else:
			Array of all results for each stock
	"""
	def build_data(self, stocks, start_date = None, end_date = None):
		self.stocks = stocks
		with open("stock_data/invalid_stocks.txt", "r+") as f:
			self.invalid_stock_codes = [line.strip() for line in f]
		
		output = [{}] * len(stocks)

		for stock_idx, stock_code in enumerate(stocks):
			if stock_code in self.invalid_stock_codes:
				print("%s is unavailable, removing..." % stock_code)
				output[stock_idx] = None

			print(stock_code)
			sec = stock_code.split("/")[1]	# Just the ticker, not the DB code
			pickle_name = self._build_pickle_name(sec, start_date, end_date)

			# If there's already a saved pickle for this data
			if os.path.isfile("./stock_data/" + pickle_name + "_data.pickle"):
				_data = pickle.load(open("./stock_data/" + pickle_name + "_data.pickle", "rb"))

				# Remove output data based on init args
				data_keys = ["trX", "trY", "testX", "testY", "price", "X", "Y"]
				if self.mode == "test":
					data_keys = [e for e in data_keys if e != "Y"]
				if not self.test:
					data_keys = [e for e in data_keys if e not in ("trX", "trY", 
																   "testX", "testY")]

				for key in data_keys:
					output[stock_idx][key] = _data[key]

			else:
				# print("No pickle found, getting data for", sec)
				try:
					# print("Getting data for", stock_code)
					df = quandl.get(stock_code, start_date = start_date,
									end_date = end_date)

				except quandl.errors.quandl_error.NotFoundError:
					# Log as faulty code
					self._log_faulty_code(stock_code)
					output[stock_idx] = None
					continue

				# Trim column names to standard O-H-L-C-V
				df = _rename_columns(df)	
				df.reset_index(drop=True, inplace = True)

				# Actually build all the technical indicators
				df = self._build_indicators(df)

				price = df["close"].values

				# If target Y data is requested, build it
				if self.mode == "train":
					Y = _compute_Y(price)

				if isinstance(price, np.ndarray):
					price = price.tolist()

				broken = False
				for idx, val in reversed(list(df.isnull().any(axis=1).iteritems())):
					if val == True:
						df.drop(idx, inplace = True)
						if self.mode == "train":
							Y.drop(idx, inplace = True)
						try:
							price.pop(idx)
						except IndexError:	#drop the security
							# print("Error, dropping security", sec)
							broken = True
							break
				if broken:
					self._log_faulty_code(stock_code)
					output[stock_idx] = None
					continue

				X = df.values
				for i, a in enumerate(X): X[i] = np.array(X[i])

				if self.mode == "train":
					Y = np.vstack(Y.values)

				# Normalize values if requested
				if not self.raw:
					if not os.path.isfile("./stock_data/" + pickle_name + ".scaler"):
						scaler = prep.StandardScaler().fit(X)
						X = scaler.transform(X)
						joblib.dump(scaler, "./stock_data/" + pickle_name + ".scaler")
					else:
						scaler = joblib.load("./stock_data/" + pickle_name + ".scaler")
						X = scaler.transform(X)

				# Make test data if requested
				if self.test:
					if self.random_split:
						trX, testX, trY, testY = train_test_split(X, Y, 
												 test_size = self.test_split, 
												 random_state=0)

					else: 		# just clips the test data off the end
						l = len(X)
						test_size = int(self.test_split*l)
						trX, testX = X[:-test_size], X[-test_size:]
						trY, testY = Y[:-test_size], Y[-test_size:]

					output[stock_idx]["trX"], output[stock_idx]["testX"] = trX, testX
					output[stock_idx]["trY"], output[stock_idx]["testY"] = trY, testY

				output[stock_idx]["X"] = X
				if self.mode == "train":
					output[stock_idx]["Y"] = Y
				output[stock_idx]["price"] = price

				pickle.dump(output[stock_idx], open("./stock_data/" + pickle_name + "_data.pickle", "wb"))
				
		return output


	def _build_indicators(self, df):
		if not self.realtime:
			inputs = df.to_dict(orient="list")
			for col in inputs:
				inputs[col] = np.array(inputs[col])

			c = df["close"]
			for n in range(2, 40):
				inputs["bband_u_"+str(n)], inputs["bband_m_"+str(n)], inputs["bband_l_"+str(n)] = ta.BBANDS(inputs, n)
				inputs["sma_"+str(n)] = ta.SMA(inputs, timeperiod = n)
				inputs["adx_"+str(n)] = ta.ADX(inputs, timeperiod = n)

				# fast_ema = c.ewm(span = n, adjust = False).mean()
				# slow_ema = c.ewm(span = n*2, adjust = False).mean()
				# macd1 = fast_ema - slow_ema
				# macd2 = macd1.ewm(span = int(n*2/3), adjust = False).mean()
				# macd3 = macd1 - macd2
				# inputs["macd_"+str(n)] = macd1.values
				# inputs["macdsignal_"+str(n)] = macd2.values
				# inputs["macdhist_"+str(n)] = macd3.values
				if n != 2:
					inputs["macd_"+str(n)], inputs["macdsignal_"+str(n)], inputs["macdhist_"+str(n)] = ta.MACD(inputs, n, n*2, int(n*2/3))
				else:
					inputs["macd_"+str(n)], inputs["macdsignal_"+str(n)], inputs["macdhist_"+str(n)] = ta.MACD(inputs, n, n*2, 1)

				# macd = [macd1.values, macd2.values, macd3.values]
				# for idx, i in enumerate(["macd_"+str(n), "macdsignal_"+str(n), "macdhist_"+str(n)]):
				# 	for day in zip(inputs[i], macd[idx]):
				# 		print("Type: %s N: %d PD: %.3f TA: %.3f, " % (i, n, day[1], day[0]))
				inputs["mfi_"+str(n)] = ta.MFI(inputs, n)
				inputs["ult_"+str(n)] = ta.ULTOSC(inputs, n, n*2, n*4)
				inputs["willr_"+str(n)] = ta.WILLR(inputs, n)
				inputs["slowk"], inputs["slowd"] = ta.STOCH(inputs)
				inputs["mom_"+str(n)] = ta.MOM(inputs, n)

			inputs["volume"] = list(map(lambda x: x/10000, inputs["volume"]))
			df = pd.DataFrame().from_dict(inputs)
			# df = df.ix[100:]

			# print(df.tail(5)["macd_3"], df.tail(5)["macdsignal_3"], df.tail(5)["macdhist_3"])			
			return df

		else:
			# Build data one-by-one, as if it's coming in one at a time
			output = pd.DataFrame()
			sliding_window = pd.DataFrame()

			for idx, day in df.iterrows():
				print("\rNow building day", str(idx), end="", flush=True)
				day = copy.deepcopy(day)	# Avoid reference vs copy bullshit
				sliding_window = sliding_window.append(day, ignore_index=True)
				# print(day, type(day))
				day_out = {}

				# print(sliding_window)
				o = sliding_window["open"].values
				h = sliding_window["high"].values
				l = sliding_window["low"].values
				c_series = sliding_window["close"]
				c = sliding_window["close"].values
				# print("----")
				# print(c)
				v = sliding_window["volume"].values

				for t in ["open", "high", "low", "close"]:
					day_out[t] = sliding_window[t].values[-1]

				for n in range(2, 40):
					# time.sleep(0.1)
					day_out["bband_u_"+str(n)], day_out["bband_m_"+str(n)], day_out["bband_l_"+str(n)] = stream.BBANDS(c, n)
					day_out["sma_"+str(n)] = stream.SMA(c, timeperiod = n)
					day_out["adx_"+str(n)] = stream.ADX(h, l, c, timeperiod = n)

					fast_ema = c_series.ewm(span = n, adjust = False).mean()
					slow_ema = c_series.ewm(span = n*2, adjust = False).mean()
					macd1 = fast_ema - slow_ema
					macd2 = macd1.ewm(span = int(n*2/3), adjust = False).mean()
					macd3 = macd1 - macd2
					day_out["macd_"+str(n)] = macd1.values[-1]
					day_out["macdsignal_"+str(n)] = macd2.values[-1]
					day_out["macdhist_"+str(n)] = macd3.values[-1]
					# if n != 2:
					# 	day_out["macd_"+str(n)], day_out["macdsignal_"+str(n)], day_out["macdhist_"+str(n)] = stream.MACD(c, n, n*2, int(n*2/3))
					# elif idx > 100:
					# 	macd =  ta.MACD({"close":c}, n, n*2, 1)
					# 	day_out["macd_2"], day_out["macdsignal_2"], day_out["macdhist_2"] = (x[-1] for x in macd)
					# else: 
					# 	day_out["macd_2"], day_out["macdsignal_2"], day_out["macdhist_2"] = None, None, None

					# macd = [macd1.values, macd2.values, macd3.values]
					# for idx, i in enumerate(["macd_"+str(n), "macdsignal_"+str(n), "macdhist_"+str(n)]):
					# 	for day in zip(inputs[i], macd[idx]):
					# 		print("Type: %s N: %d PD: %.3f TA: %.3f, " % (i, n, day[1], day[0]))
					day_out["mfi_"+str(n)] = stream.MFI(h, l, c, v, n)
					day_out["ult_"+str(n)] = stream.ULTOSC(h, l, c, n, n*2, n*4)
					day_out["willr_"+str(n)] = stream.WILLR(h, l, c, n)
					day_out["slowk"], day_out["slowd"] = stream.STOCH(h, l, c)
					day_out["mom_"+str(n)] = stream.MOM(c, n)

				day_out["volume"] = v[-1] / 10000
				# print(day_out["macd_2"], day_out["macdsignal_2"], day_out["macdhist_2"])

				output = output.append(day_out, ignore_index=True)

			# print(output.tail(5)["macd_3"], output.tail(5)["macdsignal_3"], output.tail(5)["macdhist_3"])
			return output

	def _log_faulty_code(self, stock_code):
		self.invalid_stock_codes += [stock_code]
		with open("stock_data/invalid_stocks.txt", "a") as f:
			f.write(stock_code + "\n")

	def _build_pickle_name(self, sec, start_date, end_date):
		pickle_name = sec
		if self.raw:
			pickle_name += "_raw"
		if not self.random_split:
			pickle_name += "_notrand"
		if self.test:
			pickle_name += "_withtest"
		if self.realtime:
			pickle_name += "_realtime"
		if start_date and end_date:
			pickle_name += start_date + "to" + end_date
		elif start_date:
			pickle_name += start_date
		elif end_date:
			pickle_name += "to" + end_date
		return pickle_name




def _compute_Y(price):
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

	return Y

def _rename_columns(df):

	df = copy.deepcopy(df)
	if isinstance(df, pd.core.series.Series):
		df = pd.DataFrame([df], columns = df.index.values)

	if "Adj. Close" in df.columns:
		df = df[["Adj. Open",  "Adj. High",  "Adj. Low",  "Adj. Close", "Adj. Volume"]]
		df.rename(columns=lambda x: x[5:].lower(), inplace=True)    # Remove the "Adj. " and make lowercase
	elif "Close" in df.columns:
		df = df[["Open",  "High",  "Low",  "Close", "Volume"]]
		df.rename(columns=lambda x: x.lower(), inplace=True)    # make lowercase

	return df


		