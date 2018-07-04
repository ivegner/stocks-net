'''Utility for training LSTM-type models'''
import click
import numpy as np
from keras.layers import GRU, Dense, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from callbacks import TrainValTensorBoard

from data_provider import build_train_data, get_num_features

REG_CONSTANT = 0.01

@click.command()
@click.option('-n', "--model_name", required=True, type=str)
@click.option('-e', '--n_epochs', default=50, type=int)
@click.option('-l', '--n_layers', default=3, type=int)
@click.option('-u', '--units_in_layer', default=500, type=int)
@click.option('--load', '--load_filename', default=None, type=str)
@click.option('--test_sec', default='WIKI/CZR', type=str)
@click.argument('securities', nargs=-1)
def train_model(model_name, n_epochs, n_layers, units_in_layer, load_filename, test_sec, securities):
    '''Trains model according to given CLI params'''

    if load_filename is not None:
        model = load_model(load_filename)
        last_epoch = int(load_filename.split("-")[1][0:2])
        initial_epoch = last_epoch + 1
    else:
        model = build_model(n_layers, units_in_layer)
        initial_epoch = 0


    test_data = build_train_data(test_sec, start_date="2014-01-01",
                                           end_date="2017-01-01")
    test_x, test_y = reshape_x_y(test_data['X'].as_matrix(), test_data['Y'])

    model.fit_generator(data_generator(securities),
                        steps_per_epoch=len(securities) if len(securities) < 20 else 20,
                        nb_epoch=n_epochs,
                        verbose=2,
                        callbacks=[
                            TrainValTensorBoard(model_name),
                            ModelCheckpoint(model_name + "-{epoch:02d}.h5", period=5)],
                        initial_epoch=initial_epoch,
                        validation_data=(test_x, test_y))

    scores = model.evaluate(test_x, test_y, verbose=2)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    model.save(model_name + ".h5")

def data_generator(securities):
    '''Generates data for keras in the proper format'''
    while 1:
        for sec in securities:
            data = build_train_data(sec, end_date="2017-12-01")
            # x = np.random.rand(9000, 94)
            # y = np.random.choice([0, 1], (9000,))
            if data is not None:
                out = reshape_x_y(data["X"].as_matrix(), data['Y'])
                yield out
            # yield reshape_x_y(x, y)

def build_model(n_layers, num_in_layer):
    '''Builds LSTM model as per specifications'''
    one_input_length = get_num_features()
    model = Sequential()
    model.add(GRU(num_in_layer, input_shape=(None, one_input_length), return_sequences=True,
                  W_regularizer=l2(REG_CONSTANT), U_regularizer=l2(REG_CONSTANT)))
    for i in range(n_layers-1):
        model.add(BatchNormalization())
        model.add(GRU(num_in_layer, return_sequences=True,
                      W_regularizer=l2(REG_CONSTANT), U_regularizer=l2(REG_CONSTANT)))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def reshape_x_y(x, y):
    '''Reshapes given x and y to one-batch Keras vectors'''
    r_x = np.reshape(x, (1,) + np.shape(x))
    r_y = np.reshape(y, (1,) + np.shape(y) + (1,))
    return r_x, r_y

if __name__ == '__main__':
    train_model()     #pylint:disable=E1120
