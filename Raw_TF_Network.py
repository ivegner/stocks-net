import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import dill as pickle
from Stock_NN_Funcs import build_data
import sys

np.set_printoptions(precision = 6)

'''  CONSTANTS '''

PICKLE_NAME = "_".join(s[5:] for s in sys.argv[1:])
TEST_CASH = 10000.0
# NUM_GENS = int(sys.argv[2])

''''''''''''''''''
_data = build_data(sys.argv[1:])
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
            layers.append({"weights":tf.Variable(tf.random_normal([size, layer_sizes[i+1]]), name = "weights_"+str(i)),
                           "biases":tf.Variable(tf.random_normal([layer_sizes[i+1]]), name = "biases_"+str(i))})

            if i == 0:   # the first layer
                layers[0]["output"] = tf.add(tf.matmul(data, layers[0]["weights"]), layers[0]["biases"])
                layers[0]["output"] = tf.nn.sigmoid(layers[0]["output"])
            else:
                layers[i]["output"] = tf.add(tf.matmul(layers[i-1]["output"], layers[i]["weights"]), layers[i]["biases"])

                if i != len(layer_sizes) - 2:    # Apply sigmoid if it's not the last layer
                    layers[i]["output"] = tf.nn.sigmoid(layers[i]["output"])

    return layers[-1]["output"]

best = 0
def test_trading():
    CASH = TEST_CASH
    MARGIN_CASH = 10000
    shares = 0
    flag = 0
    short_price = 0
    output = tf.argmax(prediction, 1).eval({x:X_norm[-500:]}, session = sess)
    print(output)
    # print(price[:30])
    for day_price, bar in zip(price, output):
        if bar == 2:    #buy
            # # print("buy")
            if flag == 0:       #no position
                shares = CASH / day_price
                CASH -= shares * day_price 
                flag = 1

            if flag == -1:    #short
                CASH += shares * (short_price - day_price)
                shares = 0
                flag = 0

        elif bar == 0:    #sell
            # print("sell")
            if flag == 0:       # no position
                shares = MARGIN_CASH / day_price
                short_price = day_price
                flag = -1

            elif flag == 1:    # long
                CASH += shares * day_price
                shares = 0
                flag = 0

    CASH += shares * day_price
    if CASH > best:
        global best 
        best = CASH
    saver.save(sess, "./"+PICKLE_NAME+".ckpt")

    print(CASH)


sess = tf.Session()

# def train_neural_network(x):
print("Training...")
prediction = neural_network_model(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

hm_epochs = 100
# with tf.Session() as sess:
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()

for epoch in range(hm_epochs):
    _, c = sess.run([optimizer, cost], feed_dict={x: trX[:500], y: trY[:500]})  #sets session placeholders to actual values

    print("Epoch", epoch, "completed out of", hm_epochs, "loss:", c)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    test_trading()
    # print(sess.run(prediction, feed_dict={x: trX, y: trY}))   #debug, to see outputs of prediction

    accuracy = tf.reduce_mean(tf.cast(correct, "float"))
    print("Accuracy:",accuracy.eval({x:testX, y:testY}, session = sess))

print("BEST RESULT: $", best, " from an initial investment of $", TEST_CASH)
sess.close()






