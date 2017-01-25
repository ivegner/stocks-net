import tensorflow as tf
import quandl
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import sys, os
import dill as pickle
from sklearn import preprocessing as prep
from sklearn.model_selection import train_test_split
from talib import abstract as ta

quandl.ApiConfig.api_key = "KDH1TFmmmcrjgynvRdWg"

HI_LO_DIFF = 0.03
MIN_MAX_PERIOD = 8

def build_data(raw = False):
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

		if not os.path.isfile("./stock_data/" + sec + "_data.pickle"):
			# print("No pickle found, getting data for", sec)
			try:
				# print("Getting data for", stock_code)
				df = quandl.get(stock_code)
			except quandl.errors.quandl_error.NotFoundError:
				invalid_stock_codes += [stock_code]
				f.write(stock_code + "\n")
				stock_code = yield None
				continue

			if "Adj. Close" in df.columns:
				df = df[["Adj. Open",  "Adj. High",  "Adj. Low",  "Adj. Close", "Adj. Volume"]]
				df.rename(columns=lambda x: x[5:].lower(), inplace=True)    # Remove the "Adj. " and make lowercase
			elif "Close" in df.columns:
				df = df[["Open",  "High",  "Low",  "Close", "Volume"]]
				df.rename(columns=lambda x: x.lower(), inplace=True)    # make lowercase

			price = df['close'].values
			minIdxs = argrelextrema(price, np.less)
			maxIdxs = argrelextrema(price, np.greater)


			Y = pd.Series(name="signal", dtype=np.ndarray, index=range(0, len(price)))
			n=0
			for _, idx in np.ndenumerate(minIdxs):
				# if idx < MIN_MAX_PERIOD: continue
				max_price = max(price[idx: idx + MIN_MAX_PERIOD])
				if ((max_price - price[idx]) / price[idx]) > HI_LO_DIFF:    #if the difference between max and min is > X%
					Y.set_value(idx, np.array([1., 0.], np.float32))
					n+=1

			# print("MINS:", n)
			n=0
			for _, idx in np.ndenumerate(maxIdxs):
				# if idx < MIN_MAX_PERIOD: continue
				min_price = min(price[idx: idx + MIN_MAX_PERIOD])
				if ((price[idx] - min_price)/ min_price) > HI_LO_DIFF:  #if the difference between max and min is > X%
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
						s = np.array([0., 0.]) 	# no action taken, only occurs at the beginnings of datasets, afaik

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

			for n in range(2, 40):
				inputs["bband_u_"+str(n)], inputs["bband_m_"+str(n)], inputs["bband_l_"+str(n)] = ta.BBANDS(inputs, n)
				inputs["sma_"+str(n)] = ta.SMA(inputs, timeperiod = n)
				inputs["adx_"+str(n)] = ta.ADX(inputs, timeperiod = n)
				inputs["macd_"+str(n)], inputs["macdsignal_"+str(n)], inputs["macdhist_"+str(n)] = ta.MACD(inputs, n, n*2, n*2/3)
				inputs["mfi_"+str(n)] = ta.MFI(inputs, n)
				inputs["ult_"+str(n)] = ta.ULTOSC(inputs, n, n*2, n*4)
				inputs["willr_"+str(n)] = ta.WILLR(inputs, n)
				inputs["slowk"], inputs["slowd"] = ta.STOCH(inputs)
				inputs["mom_"+str(n)] = ta.MOM(inputs, n)
				inputs["mom_"+str(n)] = ta.MOM(inputs, n)

			inputs["volume"] = list(map(lambda x: x/10000, inputs["volume"]))

			df = pd.DataFrame().from_dict(inputs)
			broken = False
			for idx, val in df.isnull().any(axis=1).iteritems():
				if val == True:
					df.drop(idx, inplace = True)
					Y.drop(idx, inplace = True)
					try:
						price.pop(idx)
					except IndexError:	#drop the security
						# print("Error, dropping security", sec)
						broken = True
						break

			''' BUILD NEURAL NET INPUTS '''
			if not broken:
				Y = np.vstack(Y.values)[20:]
				X = df.values[20:]

				if not raw:
					scaler = prep.StandardScaler().fit(X)
					X_norm = scaler.transform(X)
					from sklearn.externals import joblib
					joblib.dump(scaler, "./stock_data/" + sec + ".scaler") 
				else:
					X_norm = X

				trX, testX, trY, testY= train_test_split(X_norm, Y, test_size = 0.1, random_state=0)
				# print("Pickling...")
				output = {"X_norm": X_norm, "Y": Y, "trX": trX, "trY": trY, "testX": testX, "testY": testY, "price": price}
				pickle.dump(output, open("./stock_data/" + (sec if not raw else sec + "_raw") + "_data.pickle", "wb"))
				stock_code = yield output
			else:
				invalid_stock_codes += [stock_code]
				f.write(stock_code + "\n")
				stock_code = yield None


		else:
			# print("Pickle found, loading...")
			_data = pickle.load(open("./stock_data/" + (sec if not raw else sec + "_raw") + "_data.pickle", "rb"))
			trX, trY, testX, testY, price, X_norm, Y = _data["trX"], _data["trY"], _data["testX"], _data["testY"], _data["price"], _data["X_norm"], _data["Y"]
			stock_code = yield {"X_norm": X_norm, "Y": Y, "trX": trX, "trY": trY, "testX": testX, "testY": testY, "price": price}



def build_data_to_dict(secs, raw = False):

	PICKLE_NAME = "_".join(s[5:] for s in secs)
	print("SECURITIES: ", PICKLE_NAME.split("_"))

	if not os.path.isfile("./stock_data/" + PICKLE_NAME + "_data.pickle"):
		print("No pickle found, getting data...")
		# df = pd.concat([quandl.get("WIKI/AAPL"), quandl.get("WIKI/F"), quandl.get("WIKI/XOM")])
		df = pd.DataFrame()
		Y = pd.Series()
		prices = []
		for sec in secs:
			sec_df = quandl.get(sec)

			if "Adj. Close" in sec_df.columns:
				sec_df = sec_df[["Adj. Open",  "Adj. High",  "Adj. Low",  "Adj. Close", "Adj. Volume"]]
				sec_df.rename(columns=lambda x: x[5:].lower(), inplace=True)    # Remove the "Adj. " and make lowercase
			elif "Close" in sec_df.columns:
				sec_df = sec_df[["Open",  "High",  "Low",  "Close", "Volume"]]
				sec_df.rename(columns=lambda x: x.lower(), inplace=True)    # make lowercase

			print("Calculating output for", sec)
			price = sec_df['close'].values
			minIdxs = argrelextrema(price, np.less)
			maxIdxs = argrelextrema(price, np.greater)


			sec_Y = pd.Series(name="signal", dtype=np.ndarray, index=range(0, len(price)))
			n=0
			for _, idx in np.ndenumerate(minIdxs):
				if idx < MIN_MAX_PERIOD: continue
				max_price = max(price[idx: idx + MIN_MAX_PERIOD])
				if ((max_price - price[idx]) / price[idx]) > HI_LO_DIFF:    #if the difference between max and min is > 2%
					sec_Y.set_value(idx, np.array([1, 0, 0], np.int32))
					n+=1

			print("MINS:", n)
			n=0
			for _, idx in np.ndenumerate(maxIdxs):
				if idx < MIN_MAX_PERIOD: continue
				min_price = min(price[idx: idx + MIN_MAX_PERIOD])
				if ((price[idx] - min_price)/ min_price) > HI_LO_DIFF:  #if the difference between max and min is > 2%
					sec_Y.set_value(idx, np.array([0, 0, 1], np.int32))
					n+=1
			print("MAXS:", n)

			for idx in pd.isnull(sec_Y).nonzero()[0]:
				sec_Y.set_value(idx, np.array([0, 1, 0], np.int32))

			sec_df.reset_index(drop=True, inplace = True)
			if isinstance(price, np.ndarray):
				price = price.tolist()

			''' INDICATORS '''
			# print(len(sec_df), len(sec_Y))
			print("Building indicators...")
			inputs = sec_df.to_dict(orient="list")
			for col in inputs:
				inputs[col] = np.array(inputs[col])

			for n in range(2, 40):
				inputs["bband_u_"+str(n)], inputs["bband_m_"+str(n)], inputs["bband_l_"+str(n)] = ta.BBANDS(inputs, n)
				inputs["sma_"+str(n)] = ta.SMA(inputs, timeperiod = n)
				inputs["adx_"+str(n)] = ta.ADX(inputs, timeperiod = n)
				inputs["macd_"+str(n)], inputs["macdsignal_"+str(n)], inputs["macdhist_"+str(n)] = ta.MACD(inputs, n, n*2, n*2/3)
				inputs["mfi_"+str(n)] = ta.MFI(inputs, n)
				inputs["ult_"+str(n)] = ta.ULTOSC(inputs, n, n*2, n*4)
				inputs["willr_"+str(n)] = ta.WILLR(inputs, n)
				inputs["slowk"], inputs["slowd"] = ta.STOCH(inputs)
				inputs["mom_"+str(n)] = ta.MOM(inputs, n)
				inputs["mom_"+str(n)] = ta.MOM(inputs, n)

			inputs["volume"] = list(map(lambda x: x/10000, inputs["volume"]))

			sec_df = pd.DataFrame().from_dict(inputs)
			# print(sec_df.isnull().any(axis=1))
			for idx, val in sec_df.isnull().any(axis=1).iteritems():
				if val == True:
					# print(idx, val)
					sec_df.drop(idx, inplace = True)
					sec_Y.drop(idx, inplace = True)
					price.pop(idx)

			prices.append(price)


			df = pd.concat([df, sec_df])
			Y = pd.concat([Y, sec_Y])

		prices = [j for i in prices for j in i]	# spooky magic

		''' BUILD NEURAL NET INPUTS '''
		Y = np.vstack(Y.values)
		X = df.values

		if not raw:
			scaler = prep.StandardScaler().fit(X)
			X_norm = scaler.transform(X)
			from sklearn.externals import joblib
			joblib.dump(scaler, "./stock_data/" + sec + ".scaler") 
		else:
			X_norm = X

		trX, testX, trY, testY= train_test_split(X_norm, Y, test_size = 0.1, random_state=0)
		# print("Pickling...")
		output = {"X_norm": X_norm, "Y": Y, "trX": trX, "trY": trY, "testX": testX, "testY": testY, "price": price}
		pickle.dump(output, open("./stock_data/" + (PICKLE_NAME if not raw else PICKLE_NAME + "_raw") + "_data.pickle", "wb"))
		return output

	else:
		print("Pickle found, loading...")
		_data = pickle.load(open("./stock_data/" + PICKLE_NAME + "_data.pickle", "rb"))
		trX, trY, testX, testY, price, X_norm, Y = _data["trX"], _data["trY"], _data["testX"], _data["testY"], _data["price"], _data["X_norm"], _data["Y"]
		return {"X_norm": X_norm, "Y": Y, "trX": trX, "trY": trY, "testX": testX, "testY": testY, "price": price}



		