import easy_tensorflow as etf
import quandl
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import sys
import dill as pickle
from sklearn import preprocessing as prep

np.set_printoptions(precision = 3)
quandl.ApiConfig.api_key = "KDH1TFmmmcrjgynvRdWg"

'''  CONSTANTS '''

HI_LO_DIFF = 0.02
MIN_MAX_PERIOD = 2
SECURITY = str(sys.argv[1])
NUM_GENS = int(sys.argv[2])

''''''''''''''''''

df = quandl.get(SECURITY)

if "Adj. Close" in df.columns:
	df = df[["Adj. Open",  "Adj. High",  "Adj. Low",  "Adj. Close", "Adj. Volume"]]
	df.rename(columns=lambda x: x[5:].lower(), inplace=True)	# Remove the "Adj. " and make lowercase
elif "Close" in df.columns:
	df = df[["Open",  "High",  "Low",  "Close", "Volume"]]
	df.rename(columns=lambda x: x.lower(), inplace=True)	# make lowercase

price = df['close'].values
minIdxs = argrelextrema(price, np.less)
maxIdxs = argrelextrema(price, np.greater)

trY = pd.Series(name="signal", dtype=np.ndarray, index=range(0, len(price)))
for _, idx in np.ndenumerate(minIdxs):
	max_price = max(price[idx - MIN_MAX_PERIOD: idx + MIN_MAX_PERIOD])
	if ((max_price - price[idx]) / price[idx]) > HI_LO_DIFF:	#if the difference between max and min is > 2%
		trY.set_value(idx, np.array([1, 0, 0], np.int32))

for _, idx in np.ndenumerate(maxIdxs):
	min_price = min(price[idx - MIN_MAX_PERIOD: idx + MIN_MAX_PERIOD])
	if ((price[idx] - min_price)/ min_price) > HI_LO_DIFF:	#if the difference between max and min is > 2%
		trY.set_value(idx, np.array([0, 0, 1], np.int32))

# trY.index = df.index.values
for idx in pd.isnull(trY).nonzero()[0]:
	trY.set_value(idx, np.array([0, 0, 0], np.int32))

trY = np.vstack(trY.values)
trX = df.values
trX = prep.normalize(prep.scale(trX))   # ain't I so clever


net_type, opt, m = etf.evolve_functions.evolve('classification', 'accuracy', trX, trY, num_gens = NUM_GENS)
print(net_type, opt, m)

# trX = np.random.uniform(low=0.0, high=400.0, size=(9049,5))
# trY = np.zeros((np.size(trX, 0), 3))
# a = np.random.randint(0, 3, (np.size(trY, 0)))
# trY[np.arange(np.size(trX, 0)), a] = 1

network = etf.tf_functions.Classifier(net_type, optimizer = opt)
network.train(trX, trY, 100, return_encoded = False)

# # pickle.dump({"network": network, 
# # 			 "net_type": net_type, 
# # 			 "optimizer": opt, 
# # 			 "trX": trX, 
# # 			 "trY": trY, 
# # 			 "sym": SECURITY}, 
# # 			 open("network.pickle", "wb"))

# print("Pickled: ", net_type, opt)
# print("Accuracy: ", m)

predictions = network.predict(trX)
print(predictions(20))
# difference = set(predictions.tolist()).difference(trY.tolist())
# print(len(difference))
# print("FINAL ACCURACY: ", len(difference)/len(trX))



