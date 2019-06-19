"""Utility for training LSTM-type models"""
import click
import numpy as np
import math
import torch
from torch import nn

from dnc import DNC

from data_provider import build_train_data, get_num_features

REG_CONSTANT = 0.01
SEQ_LEN = 512
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@click.command()
@click.option("-n", "--model_name", required=True, type=str)
@click.option("-e", "--n_epochs", default=50, type=int)
@click.option("-l", "--n_layers", default=4, type=int)
@click.option("-u", "--units_in_layer", default=100, type=int)
@click.option("--load", "load_filename", default=None, type=str)
@click.option("--test_sec", default="WIKI/CZR", type=str)
@click.argument("securities", nargs=-1)
def train_model(
    model_name, n_epochs, n_layers, units_in_layer, load_filename, test_sec, securities
):
    """Trains model according to given CLI params"""

    if load_filename is not None:
        raise NotImplementedError()
        # model = load_model(load_filename)
        # last_epoch = int(load_filename.split("-")[1][0:2])
        # initial_epoch = last_epoch + 1
    else:
        model = build_model(n_layers, units_in_layer)
        initial_epoch = 0

    model.train(True)
    test_data = build_train_data(test_sec, start_date="2014-01-01", end_date="2017-01-01")
    test_x, test_y = batch_x_y(test_data["X"].as_matrix(), test_data["Y"])

    data = data_generator(securities)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for i in range(initial_epoch, n_epochs):
        moving_loss = 0
        for sec in securities:
            train_x, train_y = next(data)
            for (batch_x, batch_y) in zip(train_x, train_y):
                x, y = (
                    torch.from_numpy(batch_x).unsqueeze(0).to(device).float(),
                    torch.from_numpy(batch_y).unsqueeze(0).to(device).float(),
                )
                # print("Input", x.shape)

                (controller_hidden, memory, read_vectors) = (None, None, None)

                model.zero_grad()

                output, (controller_hidden, memory, read_vectors) = model[0](
                    x, (controller_hidden, memory, read_vectors), reset_experience=True
                )
                output = model[1](output)

                loss = criterion(output, y)
                loss.backward()

                correct = output.detach().argmax(1).float() == y
                correct = correct.clone().sum()

                if moving_loss == 0:
                    moving_loss = correct
                else:
                    moving_loss = moving_loss * 0.99 + correct * 0.01
                print("LOSS", moving_loss, end="\r")

    scores = model.evaluate(test_x, test_y, verbose=2)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    model.save(model_name + ".h5")


def data_generator(securities):
    """Generates data for keras in the proper format"""
    while 1:
        for sec in securities:
            data = build_train_data(sec, end_date="2017-12-01")
            # x = np.random.rand(9000, 94)
            # y = np.random.choice([0, 1], (9000,))
            if data is not None:
                out = batch_x_y(data["X"].as_matrix(), data["Y"])
                yield out
            # yield reshape_x_y(x, y)


def build_model(n_layers, num_in_layer):
    """Builds LSTM model as per specifications"""
    one_input_length = get_num_features()
    model = nn.ModuleList([DNC(
        input_size=one_input_length,
        hidden_size=one_input_length * 2,
        rnn_type="lstm",
        num_layers=n_layers,
        nr_cells=num_in_layer,
        cell_size=32,
        read_heads=4,
        batch_first=True,
        gpu_id=-1
    ).float().to(device).float(), torch.nn.Linear(one_input_length, 1)])
    return model


def batch_x_y(x, y):
    """Reshapes given x and y to one-batch Keras vectors"""
    # print(x.shape)
    y = np.expand_dims(y, -1)
    n_chunks = len(x)//SEQ_LEN
    trim_length = SEQ_LEN * n_chunks
    r_x = np.split(x[:trim_length], n_chunks)
    # print([t.shape for t in r_x])
    r_y = np.split(y[:trim_length], n_chunks)
    # print([t.shape for t in r_y])

    assert([t.shape[:-1] for t in r_x] == [t.shape[:-1] for t in r_y])
    return r_x, r_y

if __name__ == "__main__":
    train_model()  # pylint:disable=E1120
