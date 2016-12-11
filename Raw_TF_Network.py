import tensorflow as tf
from tensorflow.python.framework import ops
import quandl
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import sys, os
import dill as pickle
from sklearn import preprocessing as prep
from sklearn.model_selection import train_test_split
from talib import abstract as ta


np.set_printoptions(precision = 6)
quandl.ApiConfig.api_key = "KDH1TFmmmcrjgynvRdWg"


'''  CONSTANTS '''

HI_LO_DIFF = 0.03
MIN_MAX_PERIOD = 3
PICKLE_NAME = "_".join(s[5:] for s in sys.argv[1:])
TEST_CASH = 10000.0
# NUM_GENS = int(sys.argv[2])

''''''''''''''''''

if not os.path.isfile(PICKLE_NAME + "_data.pickle"):
    print("No pickle found, getting data...")
    # df = pd.concat([quandl.get("WIKI/AAPL"), quandl.get("WIKI/F"), quandl.get("WIKI/XOM")])
    df = pd.DataFrame()
    Y = pd.Series()
    for sec in sys.argv[1:]:
        sec_df = quandl.get(sec)

        if "Adj. Close" in sec_df.columns:
            sec_df = sec_df[["Adj. Open",  "Adj. High",  "Adj. Low",  "Adj. Close", "Adj. Volume"]]
            sec_df.rename(columns=lambda x: x[5:].lower(), inplace=True)    # Remove the "Adj. " and make lowercase
        elif "Close" in sec_df.columns:
            sec_df = sec_df[["Open",  "High",  "Low",  "Close", "Volume"]]
            sec_df.rename(columns=lambda x: x.lower(), inplace=True)    # make lowercase

        print("Calculating output...")
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
        for idx, val in sec_df.isnull().any(axis=1).iteritems():
            if val == True:
                sec_df.drop(idx, inplace = True)
                sec_Y.drop(idx, inplace = True)

        ''' INDICATORS '''
        print("Building indicators...")
        inputs = sec_df.to_dict(orient="list")
        for col in inputs:
            inputs[col] = np.array(inputs[col])

        for n in range(2, 40):
            print(n)
            inputs["bband_u_"+str(n)], inputs["bband_m_"+str(n)], inputs["bband_l_"+str(n)] = ta.BBANDS(inputs, n)
            inputs["sma_"+str(n)] = ta.SMA(inputs, timeperiod = n)
            inputs["adx_"+str(n)] = ta.ADX(inputs, timeperiod = n)
            inputs["macd_"+str(n)], inputs["macdsignal_"+str(n)], inputs["macdhist_"+str(n)] = ta.MACD(inputs, n, n*2, n*2/3)
            inputs["mfi_"+str(n)] = ta.MFI(inputs, n)
            inputs["ult_"+str(n)] = ta.ULTOSC(inputs, n, n*2, n*4)
            inputs["willr_"+str(n)] = ta.WILLR(inputs, n)
            inputs["slowk"], inputs["slowd"] = ta.STOCH(inputs)
            inputs["mom_"+str(n)] = ta.MOM(inputs, n)
        df = pd.concat([df, pd.DataFrame().from_dict(inputs)])
        Y = pd.concat([Y, sec_Y])
        print(sec_df.head(20))

    ''' BUILD NEURAL NET INPUTS ''' # Cut beginning for all the indicators to catch up
    with pd.option_context('mode.use_inf_as_null', True):
        for idx, val in df.isnull().any(axis=1).iteritems():
            if val == True:
                df.drop(idx, inplace = True)
                Y.drop(idx, inplace = True)

    print("Normalizing inputs...")
    Y = np.vstack(Y.values)[80:]
    X = df.values[80:]
    price = price[80:]

    X_norm = prep.normalize(prep.scale(X))   # ain't I so clever
    trX, testX, trY, testY= train_test_split(X_norm, Y, test_size = 0.3, random_state=0)
    print("Pickling...")
    pickle.dump({"X_norm": X_norm, "Y": Y, "trX": trX, "trY": trY, "testX": testX, "testY": testY, "price": price}, open(PICKLE_NAME[5:] + "_data.pickle", "wb"))

else:
    print("Pickle found, loading...")
    _data = pickle.load(open(PICKLE_NAME + "_data.pickle", "rb"))
    trX, trY, testX, testY, price, X_norm, Y = _data["trX"], _data["trY"], _data["testX"], _data["testY"], _data["price"], _data["X_norm"], _data["Y"]

# ### RANDOM INPUTS
# trX = np.random.uniform(low=0.0, high=400.0, size=(9049,5))
# print(trX[0])

# trY = np.zeros((np.size(trX, 0), 3))
# a = np.random.randint(0, 3, (np.size(trY, 0)))
# trY[np.arange(np.size(trX, 0)), a] = 1
# ### END RANDOM INPUTS


layer_sizes = [len(trX[0]), 1000, 1000, 1000, 1000, 1000, 1000, 3]   # the 3 is technically not a layer (it's the output), but it's here for convenience

x = tf.placeholder("float", [None, layer_sizes[0]])
y = tf.placeholder("float")
# test_x = tf.placeholder("float", [None, layer_sizes[0]])
# test_y = tf.placeholder("float")

def neural_network_model(data):
    layers = []
    for i, size in enumerate(layer_sizes):
        if i != len(layer_sizes) - 1:    # If it's not the last element in layer_sizes (aka not the output size), give it weights and biases
            layers.append({"weights":tf.Variable(tf.random_normal([size, layer_sizes[i+1]])),
                           "biases":tf.Variable(tf.random_normal([layer_sizes[i+1]]))})

            if i == 0:   # the first layer
                layers[0]["output"] = tf.add(tf.matmul(data, layers[0]["weights"]), layers[0]["biases"])
                layers[0]["output"] = tf.nn.sigmoid(layers[0]["output"])
            else:
                layers[i]["output"] = tf.add(tf.matmul(layers[i-1]["output"], layers[i]["weights"]), layers[i]["biases"])

                if i != len(layer_sizes) - 2:    # Apply sigmoid if it's not the last layer
                    layers[i]["output"] = tf.nn.sigmoid(layers[i]["output"])

    return layers[-1]["output"]

def test_trading():
    TEST_CASH = 10000
    shares = 0
    flag = 0
    print(tf.argmax(prediction, 1).eval({x:X_norm[:500], y:Y[:500]}, session = sess))
    # print(price[:30])
    for day_price, bar in zip(price, tf.argmax(prediction, 1).eval({x:X_norm[:500], y:Y[:500]}, session = sess)):
        if bar == 2:    #buy
            # print("buy")
            if flag == 0:       #no position
                shares = TEST_CASH / day_price
                TEST_CASH -= shares * day_price 
                flag = 1

            # elif flag == -1:    #short
            #     TEST_CASH += shares * day_price
            #     shares = 0
            #     flag = 0

        elif bar == 0:    #sell
            # print("sell")
            # if flag == 0:       # no position
            #     shares = TEST_CASH / day_price
            #     TEST_CASH -= shares * day_price 
            #     flag = -1

            # elif flag == 1:    # long
            TEST_CASH += shares * day_price
            shares = 0
            flag = 0

    TEST_CASH += shares * day_price
    print(TEST_CASH)


sess = tf.Session()
# def train_neural_network(x):
print("Training...")
prediction = neural_network_model(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

hm_epochs = 50
# with tf.Session() as sess:
sess.run(tf.initialize_all_variables())

for epoch in range(hm_epochs):
    _, c = sess.run([optimizer, cost], feed_dict={x: trX, y: trY})  #sets session placeholders to actual values

    print("Epoch", epoch, "completed out of", hm_epochs, "loss:", c)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    test_trading()
    # print(sess.run(prediction, feed_dict={x: trX, y: trY}))   #debug, to see outputs of prediction

    accuracy = tf.reduce_mean(tf.cast(correct, "float"))
    print("Accuracy:",accuracy.eval({x:testX, y:testY}, session = sess))
sess.close()
