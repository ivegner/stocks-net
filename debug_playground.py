from Stock_NN_Funcs import build_realtime_data
import quandl, sys
from talib import abstract as ta
import pandas as pd
from collections import OrderedDict
import numpy as np

data = OrderedDict()

# for i in range(10):
# 	builder = build_realtime_data(["WIKI/AAPL"], start_date = "2014-01-01", end_date = "2016-01-01", backload = 200)

# 	_ = builder.send(None)

# 	for day in range(20):

# 		day_data = next(builder)
# 		if not len(data) > day:
# 			data.append(day_data)

# 		else:
# 			# print(data[day][1][0]["data"], day_data[1][0]["data"])
# 			for i in range(0, 500, 40):
# 				if not np.array_equal(data[day][1][0]["data"][i:i+40], day_data[1][0]["data"][i:i+40]):
# 					print(data[day][1][0]["data"][i:i+40])
# 					print(day_data[1][0]["data"][i:i+40])


def _build_indicators(num_secs):	# accepts a list of one-day Series

	sec_idx_range = range(num_secs)
	sliding_window = []	# list of pd.DataFrames

	data = yield

	for datum in data:
		sliding_window += [_rename_columns(datum)]

	current_day = 0
	while True:
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

				c = series.close
				for n in range(2, 40):
					inputs["bband_u_"+str(n)], inputs["bband_m_"+str(n)], inputs["bband_l_"+str(n)] = ta.BBANDS(inputs, n)
					inputs["sma_"+str(n)] = ta.SMA(inputs, timeperiod = n)
					inputs["adx_"+str(n)] = ta.ADX(inputs, timeperiod = n)
					# print("\nINPUTS:", inputs)
					# if current_day > n*2:
					fast_ema = c.ewm(span = n).mean()
					slow_ema = c.ewm(span = n*2).mean()

					# print(fast_ema, slow_ema)
					macd1 = fast_ema - slow_ema
					macd2 = macd1.ewm(span = n*2/3).mean()
					macd3 = macd1 - macd2
					inputs["macd_"+str(n)], inputs["macdsignal_"+str(n)], inputs["macdhist_"+str(n)] = macd1.iloc[-1], macd2.iloc[-1], macd3.iloc[-1]

					if current_day == 160:
						print(n)
						print(macd1, macd2, macd3)
						sys.exit(69)
					# else:
					# 	inputs["macd_"+str(n)], inputs["macdsignal_"+str(n)], inputs["macdhist_"+str(n)] = [np.NaN]*3

					inputs["mfi_"+str(n)] = ta.MFI(inputs, n)
					inputs["ult_"+str(n)] = ta.ULTOSC(inputs, n, n*2, n*4)
					inputs["willr_"+str(n)] = ta.WILLR(inputs, n)
					inputs["slowk"], inputs["slowd"] = ta.STOCH(inputs)
					inputs["mom_"+str(n)] = ta.MOM(inputs, n)
					inputs["mom_"+str(n)] = ta.MOM(inputs, n)

				inputs["volume"] = list(map(lambda x: x/10000, inputs["volume"]))

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

				if current_day < 170:
					passes_validity_check = True

				elif not np.isnan(X).any():
					passes_validity_check = True
					# if num_validation_iterations != 0:
						# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
						# 	print(series.iloc[-1])
						# 	sys.exit(1)

				else:
					num_validation_iterations += 1
					print("Reevaluating, iteration", num_validation_iterations, "day:", current_day)
					# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
					# 	print(series.iloc[-1])
					# 	sys.exit(1)

				# if current_day > 170: 
					# print(series.iloc[-1].values)


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

def array_equal(a,b):
	try:
		np.testing.assert_equal(a,b)
	except AssertionError:
		return False
	return True



def _rename_columns(df):
	if isinstance(df, pd.core.series.Series):
		df = pd.DataFrame([df], columns = df.index.values)

	if "Adj. Close" in df.columns:
		df = df[["Adj. Open",  "Adj. High",  "Adj. Low",  "Adj. Close", "Adj. Volume"]]
		df = df.rename(columns=lambda x: x[5:].lower())    # Remove the "Adj. " and make lowercase
	elif "Close" in df.columns:
		df = df[["Open",  "High",  "Low",  "Close", "Volume"]]
		df = df.rename(columns=lambda x: x.lower())    # make lowercase

	return df



for i in range(100):
	new_data = quandl.get("WIKI/AAPL", start_date = "2014-01-01", end_date = "2016-01-01")
	builder = _build_indicators(1)
	builder.send(None)

	for date, day_data in new_data.iterrows():
		if len(data.setdefault(date, [])) == 0:
			data[date] = builder.send([day_data])[0]["data"]

		else:
			x = builder.send([day_data])[0]["data"]
			if not array_equal(data[date], x):
				print("ALLAHU AKBAR")
				print(data[date], x)
				sys.exit(1)

	print("Iteration", i, "complete.")			
