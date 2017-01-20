import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import dill as pickle
from Stock_NN_Funcs import build_data
import sys
import argparse
from progress.bar import Bar

np.set_printoptions(precision = 3)

'''  CONSTANTS '''

TEST_CASH = 10000.0
load_model_arg = sys.argv[1] == "-l"
# NUM_GENS = int(sys.argv[2])

''''''''''''''''''

# ### RANDOM INPUTS
# trX = np.random.uniform(low=0.0, high=400.0, size=(9049,5))
# print(trX[0])

# trY = np.zeros((np.size(trX, 0), 3))
# a = np.random.randint(0, 3, (np.size(trY, 0)))
# trY[np.arange(np.size(trX, 0)), a] = 1
# ### END RANDOM INPUTS

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

def neural_network_model_loaded(data):
	num_layers = tf.trainable_variables()[-1][-3] + 1	#last layer number postscript + 1. Based on the naming structure of TF vars
	layers = []
	for idx, var in enumerate(tf.trainable_variables()):
		if (idx % 2): continue		#if it's an odd-numbered var, i.e. a bias
		i = int(var.name[-3])
		layers.append({"weights": var,
					   "biases": tf.trainable_variables()[idx+1]})

		# if var.name.startswith("weights") :
		# 	layers[i]["weights"] = var.eval(session = sess)
		# elif var.name.startswith("biases") :
		# 	layers[i]["biases"] = var.eval(session = sess)

		if i == 0:   # the first layer
			layers[0]["output"] = tf.add(tf.matmul(data, layers[0]["weights"]), layers[0]["biases"])
			layers[0]["output"] = tf.nn.sigmoid(layers[0]["output"])
		else:
			layers[i]["output"] = tf.add(tf.matmul(layers[i-1]["output"], layers[i]["weights"]), layers[i]["biases"])

			if i != num_layers - 1:    # Apply sigmoid if it's not the last layer
				layers[i]["output"] = tf.nn.sigmoid(layers[i]["output"])

	return layers[-1]["output"]


best = 0

def test_trading():
	CASH = TEST_CASH
	MARGIN_CASH = 10000
	shares = 0
	flag = 0
	short_price = 0
	output = tf.argmax(prediction, 1).eval({x:test_X_norm}, session = sess)
	print(output)
	# print(price[:30])
	for day_price, bar in zip(test_price, output):
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

	if flag == -1:
		CASH += shares * (short_price - day_price)
	elif flag == 1:
		CASH += shares * day_price
	if CASH > best:
		global best 
		best = CASH
	# saver.save(sess, "./"+"_".join(s.split("/")[1] for s in sys.argv[1:])+".ckpt")

	print(CASH)

print("Training...")

builder = build_data()
builder.send(None)
one_input_length = len(builder.send("WIKI/AAPL")["trX"][0])
layer_sizes = [one_input_length, 1000, 1000, 3]   # the 3 is technically not a layer (it's the output), but it's here for convenience

if not load_model_arg:
	x = tf.placeholder("float", [None, one_input_length])
	y = tf.placeholder("float")
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	saver = tf.train.Saver()

	tf.add_to_collection('train_op', optimizer)
	tf.add_to_collection('cost_op', cost)
	tf.add_to_collection('input', x)
	tf.add_to_collection('target', y)

	initial_epoch = 0

else:
	from glob import glob
	history = list(map(lambda x: int(x.split('-')[1][:-5]), glob('trained_model.ckpt-*.meta')))
	last_epoch = np.max(history)
	# Instantiate saver object using previously saved meta-graph
	saver = tf.train.import_meta_graph('trained_model.ckpt-{}.meta'.format(last_epoch))
	initial_epoch = last_epoch + 1

# if sys.argv[1] == "-l": # IF MODEL NEEDS TO BE LOADED 
# 	saver = tf.train.import_meta_graph("./massive_69.meta")
# 	saver.restore(sess, "./massive_69")
# 	prediction = neural_network_model_loaded(x)
# 	temp = set(tf.global_variables())

# elif sys.argv[1] == "-n":	# IF MODEL NEEDS TO BE MADE FROM SCRATCH
# 	prediction = neural_network_model(x)
# 	saver = tf.train.Saver()
# 	temp = set([])

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
# optimizer = tf.train.AdamOptimizer().minimize(cost)

n_epochs = 500

test_data = builder.send("WIKI/BBBY")
test_price, test_X_norm= test_data["price"], test_data["X_norm"]

#I honestly don't know how else to initialize ADAM in TensorFlow.
# uninitialized_variables = list(tf.get_variable(name) for name in
# 							   sess.run(tf.report_uninitialized_variables(tf.global_variables() + tf.local_variables())))
# sess.run(tf.variables_initializer(uninitialized_variables))

sess = tf.Session()

if not load_model_arg:
	sess.run(tf.global_variables_initializer())
else:
    saver.restore(sess, 'model.ckpt')
    optimizer = tf.get_collection('train_op')[0]
    cost = tf.get_collection('cost_op')[0]
    x = tf.get_collection('input')[0]
    y = tf.get_collection('target')[0]
    prediction = neural_network_model_loaded(x)

test_trading()

for epoch in range(n_epochs):
	c = 0

	for sec in Bar("Processing", suffix = "%(percent)d%%").iter(sys.argv[2:]):	#built-in progress bar that also iterates
		# print("Now training on", sec)
		_data = builder.send(sec)
		if _data is None:
			continue
		trX, trY, testX, testY, price, X_norm, Y = _data["trX"], _data["trY"], _data["testX"], _data["testY"], _data["price"], _data["X_norm"], _data["Y"]

		_, loss = sess.run([optimizer, cost], feed_dict={x: trX, y: trY})  #sets session placeholders to actual values
		c += loss

	print("Epoch", epoch, "completed out of", n_epochs, "loss:", c)
	# if c < 0.0001: break

	correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
	test_trading()
	# print(sess.run(prediction, feed_dict={x: trX, y: trY}))   #debug, to see outputs of prediction

	accuracy = tf.reduce_mean(tf.cast(correct, "float"))
	print("Accuracy:",accuracy.eval({x:testX, y:testY}, session = sess))

	if epoch % 3 == 0:
		saver.save(sess, 'trained_model.ckpt', global_step=epoch, keep_max = 30)

print("BEST RESULT: $", best, " from an initial investment of $", TEST_CASH)
sess.close()






