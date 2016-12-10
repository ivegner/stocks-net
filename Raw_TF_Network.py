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

HI_LO_DIFF = 0.02
MIN_MAX_PERIOD = 3
SECURITY = str(sys.argv[1])
TEST_CASH = 10000.0
# NUM_GENS = int(sys.argv[2])

''''''''''''''''''

if not os.path.isfile(SECURITY[5:] + "_data.pickle"):
    print("No pickle found, getting data...")
    # df = pd.concat([quandl.get("WIKI/AAPL"), quandl.get("WIKI/F"), quandl.get("WIKI/XOM")])
    df = quandl.get(SECURITY)

    if "Adj. Close" in df.columns:
        df = df[["Adj. Open",  "Adj. High",  "Adj. Low",  "Adj. Close", "Adj. Volume"]]
        df.rename(columns=lambda x: x[5:].lower(), inplace=True)    # Remove the "Adj. " and make lowercase
    elif "Close" in df.columns:
        df = df[["Open",  "High",  "Low",  "Close", "Volume"]]
        df.rename(columns=lambda x: x.lower(), inplace=True)    # make lowercase

    print("Calculating output...")
    price = df['close'].values
    minIdxs = argrelextrema(price, np.less)
    maxIdxs = argrelextrema(price, np.greater)

    trY = pd.Series(name="signal", dtype=np.ndarray, index=range(0, len(price)))
    for _, idx in np.ndenumerate(minIdxs):
        if idx < MIN_MAX_PERIOD: continue
        max_price = max(price[idx - MIN_MAX_PERIOD: idx + MIN_MAX_PERIOD])
        if ((max_price - price[idx]) / price[idx]) > HI_LO_DIFF:    #if the difference between max and min is > 2%
            trY.set_value(idx, np.array([1, 0, 0], np.int32))

    for _, idx in np.ndenumerate(maxIdxs):
        if idx < MIN_MAX_PERIOD: continue
        min_price = min(price[idx - MIN_MAX_PERIOD: idx + MIN_MAX_PERIOD])
        if ((price[idx] - min_price)/ min_price) > HI_LO_DIFF:  #if the difference between max and min is > 2%
            trY.set_value(idx, np.array([0, 0, 1], np.int32))

    for idx in pd.isnull(trY).nonzero()[0]:
        trY.set_value(idx, np.array([0, 1, 0], np.int32))

    df.reset_index(drop=True, inplace = True)
    for idx, val in df.isnull().any(axis=1).iteritems():
        if val == True:
            df.drop(idx, inplace = True)
            trY.drop(idx, inplace = True)

    ''' INDICATORS '''
    print("Building indicators...")
    inputs = df.to_dict(orient="list")
    for col in inputs:
        inputs[col] = np.array(inputs[col])

    for n in range(2, 20):
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

    df = df.from_dict(inputs)

    ''' BUILD NEURAL NET INPUTS ''' # Cut 20 for all the indicators to catch up
    print("Normalizing inputs...")
    trY = np.vstack(trY.values)[80:]
    trX = df.values[80:]
    price = price[80:]

    trX = prep.normalize(prep.scale(trX))   # ain't I so clever
    trX, testX, trY, testY, price, __ = train_test_split(trX, trY, price, test_size = 0.3, random_state=0)
    print("Pickling...")
    pickle.dump({"trX": trX, "trY": trY, "testX": testX, "testY": testY, "price": price}, open(SECURITY[5:] + "_data.pickle", "wb"))

else:
    print("Pickle found, loading...")
    _data = pickle.load(open(SECURITY[5:] + "_data.pickle", "rb"))
    trX, trY, testX, testY, price = _data["trX"], _data["trY"], _data["testX"], _data["testY"], _data["price"]

# ### RANDOM INPUTS
# trX = np.random.uniform(low=0.0, high=400.0, size=(9049,5))
# print(trX[0])

# trY = np.zeros((np.size(trX, 0), 3))
# a = np.random.randint(0, 3, (np.size(trY, 0)))
# trY[np.arange(np.size(trX, 0)), a] = 1
# ### END RANDOM INPUTS


layer_sizes = [len(trX[0]), 1000, 1000, 1000, 1000, 1000, 3]   # the 3 is technically not a layer (it's the output), but it's here for convenience

x = tf.placeholder("float", [None, layer_sizes[0]])
y = tf.placeholder("float")
# test_x = tf.placeholder("float", [None, layer_sizes[0]])
# test_y = tf.placeholder("float")

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):  #I've no idea what this is
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def trade_cost_func(x, name=None):
    with ops.name_scope(name, "TradeCost", [x]) as name:
        cost = py_func(cost_function,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=cost_grad)
        return cost[0]

def cost_grad(op, grad):
    return tf.reshape(tf.convert_to_tensor([0.0, 0.0, 0.0]), (1, 3))

def cost_function(predicted):
    cash = TEST_CASH
    shares = 0
    flag = 0
    print("test0")
    print(predicted.get_shape())
    for day_price, bar in zip(price, tf.unpack(predicted, 0)):
        print("test1")
        if bar == [0, 0, 1]:    #buy
            if flag == 0:       #no position
                print("buy")
                shares = cash / day_price
                cash -= shares * day_price 
                flag = 1

            elif flag == -1:    #short
                cash += shares * day_price
                shares = 0
                flag = 0

        elif bar == [1, 0, 0]:    #sell
            if flag == 0:       # no position
                shares = cash / day_price
                cash -= shares * day_price 
                flag = -1

            elif flag == 1:    # long
                cash += shares * day_price
                shares = 0
                flag = 0
    print("return")
    return tf.to_float(1/cash)

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

    return tf.one_hot(tf.argmax(layers[-1]["output"], 1), depth = 3, axis = -1)

def train_neural_network(x):
    print("Training...")
    prediction = neural_network_model(x)
    cost = trade_cost_func(prediction)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 20
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            _, c = sess.run([optimizer, cost], feed_dict={x: trX, y: trY})  #sets session placeholders to actual values

            print("Epoch", epoch, "completed out of", hm_epochs, "loss:", c)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            # print(sess.run(prediction, feed_dict={x: trX, y: trY}))   #debug, to see outputs of prediction

            accuracy = tf.reduce_mean(tf.cast(correct, "float"))
            print("Accuracy:",accuracy.eval({x:testX, y:testY}))

train_neural_network(x)
