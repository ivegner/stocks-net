import sys, os

# I welcome all suggestions for how to do this better
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from keras.models import Sequential
from keras.layers import TimeDistributed, Dense, GRU
from keras.regularizers import l2, l1
from keras.callbacks import Callback, ModelCheckpoint
from data_provider import DataBuilder

np.random.seed(7)

load_model_arg = sys.argv[1] == "-l"

train_secs = sys.argv[(3 if load_model_arg else 1) :]

samples_per_epoch = len(train_secs)
nb_epoch = 50

data_builder = DataBuilder(random_split=False, test=True, realtime=True, mode="train")
aapl = data_builder.build_data(["WIKI/AAPL"], start_date="2014-01-01", end_date="2017-01-01")[0]
one_input_length = len(aapl["X"][0])

print(train_secs)


def keras_builder():
    # s = yield
    while 1:
        for sec in train_secs:
            data = data_builder.build_data([sec], end_date="2016-01-01")[0]
            if data is not None:
                x = np.reshape(data["trX"], (1,) + np.shape(data["trX"]))
                y = np.reshape(data["trY"], (1,) + np.shape(data["trY"]))
                # for x, y in zip(x, y)
                yield x, y


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print("\nTesting loss: {0:.2f}, acc: {1:.2f}\n".format(loss, acc))


if not load_model_arg:
    model = Sequential()
    model.add(
        GRU(
            1000,
            input_dim=one_input_length,
            return_sequences=True,
            W_regularizer=l2(0.1),
            U_regularizer=l2(0.1),
        )
    )
    model.add(GRU(1000, return_sequences=True, W_regularizer=l2(0.1), U_regularizer=l2(0.1)))
    model.add(GRU(1000, return_sequences=True, W_regularizer=l2(0.1), U_regularizer=l2(0.1)))
    model.add(TimeDistributed(Dense(2, activation="sigmoid")))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    initial_epoch = 0
else:
    from keras.models import load_model

    name = sys.argv[2]
    last_epoch = int(name.split("-")[1][0:2])
    model = load_model(name)
    initial_epoch = last_epoch + 1

print(model.summary())

test_X = np.reshape(aapl["testX"], (1,) + np.shape(aapl["testX"]))
test_Y = np.reshape(aapl["testY"], (1,) + np.shape(aapl["testY"]))

model.fit_generator(
    keras_builder(),
    samples_per_epoch=samples_per_epoch,
    nb_epoch=nb_epoch,
    verbose=2,
    callbacks=[
        TestCallback((test_X, test_Y)),
        ModelCheckpoint("lstm_model-{epoch:02d}.h5", period=2),
    ],
    initial_epoch=initial_epoch,
)

scores = model.evaluate(test_X, test_Y, verbose=2)
print("Accuracy: %.2f%%" % (scores[1] * 100))

model.save("lstm_model.h5")
