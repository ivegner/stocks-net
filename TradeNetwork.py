import tensorflow as tf
import numpy as np
import dill as pickle
from Stock_NN_Funcs import build_data


TEST_CASH = 10000

PICKLE_NAME = "_".join(s[5:] for s in sys.argv[1:])

sess = tf.Session()
new_saver = tf.train.import_meta_graph("./" + PICKLE_NAME + ".ckpt.meta")
new_saver.restore(sess, "./" + PICKLE_NAME + ".ckpt")
# all_vars = tf.trainable_variables()
# for v in all_vars:
#     print(v.name)
#     print(len(v.eval(session = sess)))


def neural_network_model(data):
	num_layers = tf.trainable_variables()[-1][-3] + 1	#last layer number postscript + 1. Based on the naming structure of TF vars
	layers = []
	for idx, var in enumerate(tf.trainable_variables()):
		if (idx % 2): continue		#if it's an odd-numbered var, i.e. a bias
		i = int(var.name[-3])
		print(i)
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
			print("I: ", i)

			if i != num_layers - 1:    # Apply sigmoid if it's not the last layer
				layers[i]["output"] = tf.nn.sigmoid(layers[i]["output"])

	print("Length: ", len(layers))
	return layers[-1]["output"]

_data = build_data(["WIKI/BBBY"])
price, X_norm= _data["price"], _data["X_norm"]
print("Total lengths: ", len(price), len(X_norm))

x = tf.placeholder("float", [None, len(X_norm[0])])
y = tf.placeholder("float")

prediction = neural_network_model(x)

best = 0
chunk = 1

for prices, data in zip([price[i:i+252] for i in range(0, len(price), 252)], [X_norm[i:i+252] for i in range(0, len(X_norm), 252)]):
	chunk+=1
	CASH = TEST_CASH
	MARGIN_CASH = 10000
	shares = 0
	flag = 0
	short_price = 0
	output = tf.argmax(prediction, 1).eval({x:data}, session = sess)
	print(output)
	# print(price[:30])
	print("Batch length: ", len(prices), len(output))

	for day_price, bar in zip(prices, output):
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
		# saver.save(sess, "./"+PICKLE_NAME+".ckpt")

	print("Year ", chunk, " returned $", CASH, " from an investment of $", TEST_CASH)

sess.close()

'''
NETWORK TEST CODE 
'''
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
# optimizer = tf.train.AdamOptimizer().minimize(cost)

# hm_epochs = 2
# # with tf.Session() as sess:
# sess.run(tf.initialize_all_variables())

# saver = tf.train.Saver()

# for epoch in range(hm_epochs):
# 	_, c = sess.run([optimizer, cost], feed_dict={x: trX[:500], y: trY[:500]})  #sets session placeholders to actual values

# 	print("Epoch", epoch, "completed out of", hm_epochs, "loss:", c)

# 	correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
# 	# test_trading()
# 	# print(sess.run(prediction, feed_dict={x: trX, y: trY}))   #debug, to see outputs of prediction

# 	accuracy = tf.reduce_mean(tf.cast(correct, "float"))
# 	print("Accuracy:",accuracy.eval({x:testX, y:testY}, session = sess))


