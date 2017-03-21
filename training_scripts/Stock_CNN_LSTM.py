import sys, os
# I welcome all suggestions for how to do this better
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))	 

import numpy as np
from keras.models import Sequential, Model
from keras.layers import TimeDistributed, Dense, GRU, Convolution1D, MaxPooling1D, Lambda
from keras.regularizers import l2, l1
from keras.callbacks import Callback, ModelCheckpoint
from Stock_NN_Funcs import build_data
from keras import backend as K

np.random.seed(7)

load_model_arg = sys.argv[1] == "-l"

train_secs = sys.argv[(2 if load_model_arg else 1):]

samples_per_epoch = len(train_secs)
nb_epoch = 120

_build_data = build_data(random_split = False)
_build_data.send(None)
aapl = _build_data.send("WIKI/AAPL")
one_input_length = len(aapl["trX"][0])

def keras_builder(builder):
	# s = yield
	while(1):
		for stock_code in train_secs:
			output = builder.send(stock_code)
			if output is not None:
				x = np.reshape(output["trX"], (1,) + np.shape(output["trX"]))
				y = np.reshape(output["trY"], (1,) + np.shape(output["trY"]))
				# for x, y in zip(x, y)
				yield x, y

class TestCallback(Callback):
	def __init__(self, test_data):
		self.test_data = test_data

	def on_epoch_end(self, epoch, logs={}):
		x, y = self.test_data
		loss, acc = self.model.evaluate(x, y, verbose=0)
		print('\nTesting loss: {0:.2f}, acc: {1:.2f}\n'.format(loss, acc))

test_X = np.reshape(aapl["testX"], (1,) + np.shape(aapl["testX"]))
test_Y = np.reshape(aapl["testY"], (1,) + np.shape(aapl["testY"]))

if not load_model_arg:
	model = Sequential()
	print(np.shape(test_X))
	model.add(Lambda(lambda x: K.expand_dims(x, -1), input_shape = (None, one_input_length)))
	model.add(TimeDistributed(Convolution1D(1, 5, border_mode = "same", W_regularizer = l2(0.01))))
	model.add(TimeDistributed(MaxPooling1D(3, border_mode = "same")))
	model.add(TimeDistributed(Convolution1D(1, 5, border_mode = "same", W_regularizer = l2(0.01))))
	model.add(TimeDistributed(MaxPooling1D(3, border_mode = "same")))
	model.add(Lambda(lambda x: K.squeeze(x, -1)))
	model.add(GRU(300, return_sequences = True, W_regularizer = l2(0.01), U_regularizer = l2(0.01)))
	model.add(TimeDistributed(Dense(2, activation='sigmoid')))
	# model.add(GRU(300, return_sequences = True, W_regularizer = l2(0.01), U_regularizer = l2(0.01)))

	# lambda_1_layer = Model(input=model.input,
	# 					   output=model.get_layer("lambda_1").output)
	# print(np.shape(lambda_1_layer.predict(test_X)))
	# first_layer_model = Model(input=model.input,
	# 						  output=model.get_layer("timedistributed_1").output)
	# print(np.shape(first_layer_model.predict(test_X)))
	# second_layer_model = Model(input=model.input,
	# 						   output=model.get_layer("timedistributed_2").output)
	# print(np.shape(second_layer_model.predict(test_X)))
	# layer3_model = Model(input=model.input,
	# 					 output=model.get_layer("timedistributed_3").output)
	# print(np.shape(layer3_model.predict(test_X)))
	# layer4_model = Model(input=model.input,
	# 					 output=model.get_layer("timedistributed_4").output)
	# print(np.shape(layer4_model.predict(test_X)))
	# layer5_model = Model(input=model.input,
	# 					 output=model.get_layer("lambda_2").output)
	# print(np.shape(layer5_model.predict(test_X)))
	# layer6_model = Model(input=model.input,
	# 					 output=model.get_layer("gru_1").output)
	# print(np.shape(layer6_model.predict(test_X)))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	initial_epoch = 1
else:
	from keras.models import load_model
	from glob import glob
	history = list(map(lambda x: int(x.split("-")[1][0:2]), glob("./lstm_model-*.h5")))
	last_epoch = np.max(history)
	model = load_model("lstm_model-{}.h5".format(str(last_epoch).zfill(2)))
	initial_epoch = last_epoch + 1

print(model.summary())

# test_X = np.reshape(aapl["testX"], (1,) + np.shape(aapl["testX"]))
# test_Y = np.reshape(aapl["testY"], (1,) + np.shape(aapl["testY"]))

model.fit_generator(keras_builder(_build_data), samples_per_epoch = samples_per_epoch, nb_epoch=nb_epoch, 
					verbose = 2, callbacks=[TestCallback((test_X, test_Y)), ModelCheckpoint("cnn_lstm_model-{epoch:02d}.h5")], initial_epoch = initial_epoch)

# model.fit(test_X, test_Y, nb_epoch=nb_epoch, verbose = 2, callbacks=[TestCallback((test_X, test_Y)), ModelCheckpoint("cnn_lstm_model-{epoch:02d}.h5")], initial_epoch = initial_epoch)


scores = model.evaluate(test_X, test_Y, verbose=2)
print("Accuracy: %.2f%%" % (scores[1]*100))

model.save("lstm_model.h5")

