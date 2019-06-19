import sys, os

# I welcome all suggestions for how to do this better
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Stock_NN_Funcs import build_data
import numpy as np
from keras.models import load_model, Sequential
from keras.layers import GRU, Dense, TimeDistributed
from keras.regularizers import l2, l1
import math, random

np.random.seed(94)

TEST_CASH = 1000
train_secs = ["WIKI/ORCL"]
trading_days_in_year = 252


_build_data = build_data(
    random_split=False, start_date="2003-01-01", end_date="2017-01-01", raw=True
)
_build_data.send(None)
aapl = _build_data.send("WIKI/AAPL")
one_input_length = len(aapl["trX"][0])


def data_builder(builder, train_secs):
    # s = yield
    while 1:
        for stock_code in train_secs:
            output = builder.send(stock_code)
            if output is not None:
                for day in zip(output["X_norm"], output["price"]):
                    # x = np.reshape(day[0], (1, 1,) + np.shape(day[0]))
                    x = day[0]
                    price = day[1]
                    yield {"x": x, "price": price}
            yield None


def make_trader():
    price, action = yield
    buys, sells = 0, 0

    shares = None
    bought_flag = 0  # 1 for active trade, 0 for no active trades
    bought_price = None

    cash = TEST_CASH

    while 1:  # every price-tick
        if bought_flag == 1:
            if action == 0:
                # active position and redundant signal to buy
                potential_profit_per_share = price - bought_price
                # print("REWARD:", potential_profit_per_share)
                price, action = (
                    yield potential_profit_per_share
                )  # reward = potential profit if sold
            else:
                # active position and signal to sell
                sells += 1
                total_profit = shares * (price - bought_price)
                # print("SELL:\tShares:", shares, "\tprice", round(price, 2), "\tProfit from trade:", total_profit)
                cash += total_profit
                cash = round(cash, 2)
                # print("Current cash:", cash)
                # print("REWARD PER SHARE:", total_profit/shares)
                if shares:
                    price, action = yield total_profit / shares
                else:
                    price, action = yield 0
                shares = 0
                bought_flag, bought_price = 0, None
                # plots[i, 0].plot(day_count, price + 0.2, marker = "v", color = "r", ms = 5)

        else:
            if action == 0:  # buy
                buys += 1
                # print("Long on ticker", bought_ticker)
                shares = math.floor(cash / price)
                trade_cost = shares * price
                # print("BUY:\tShares:", shares, "\tprice", round(price, 2), "\tcost:", round(trade_cost, 2), "\tRemaining cash:", round(cash - trade_cost, 2))
                bought_flag, bought_price = 1, price
                price, action = yield 0.1
                # plots[i, 0].plot(day_count, price - 0.2, marker = "^", color = "g", ms = 5)

            elif action == 1:  # Sell signal with no position
                price, action = yield -0.1

                # print("ACTION TAKEN:", action)


data_builder = data_builder(_build_data, train_secs)

model = Sequential()
model.add(Dense(500, input_dim=one_input_length, activation="relu"))
model.add(Dense(500, activation="relu"))
# model.add(GRU(500, batch_input_shape = (1, 1, one_input_length),
# 			  return_sequences = True, W_regularizer = l2(0.01),
# 			  U_regularizer = l2(0.01), activation='relu', stateful = True))
# model.add(GRU(500, return_sequences = True,
# 			  W_regularizer = l2(0.01), U_regularizer = l2(0.01),
# 			  activation='relu', stateful = True))
# model.add(GRU(500, return_sequences = True,
# 			  W_regularizer = l2(0.01), U_regularizer = l2(0.01),
# 			  activation='relu', stateful = True))
# model.add(GRU(500, return_sequences = True, W_regularizer = l2(0.01), U_regularizer = l2(0.01)))
# model.add(GRU(500, return_sequences = True, W_regularizer = l2(0.01), U_regularizer = l2(0.01)))
# model.add(TimeDistributed(Dense(2, activation='relu')))
model.add(Dense(2, activation="relu"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

epochs = 7
gamma = 0.990
epsilon = 1
batchSize = 80
buffer = 800
replay = []
# stores tuples of (S, A, R, S')
h = 0

for i in range(epochs):
    print("EPOCH", i, "epsilon", epsilon)
    for sec_idx in range(len(train_secs)):
        replay = []
        state = next(data_builder)  # dict of "x" and "price" for 1 day

        trader = make_trader()
        trader.send(None)

        # while trading still in progress
        day_count = 0
        status = 1
        while state is not None and status == 1:
            day_count += 1
            # print("Security:", train_secs[sec_idx], "day", day_count, "epsilon:", epsilon)
            x = state["x"]
            x = np.reshape(
                x, (1, one_input_length)
            )  # state is a day of shape (1, one_input_length?)
            price = state["price"]
            # print(state)
            # We are in state S
            # Let's run our Q function on S to get Q values for all possible actions

            qval = model.predict(x)
            if np.random.random() < epsilon:  # choose random action
                action = np.random.randint(0, 2)
                # print("RANDOM ACTION:", action)
            else:  # choose best action from Q(s,a) values
                action = np.argmax(qval)  # 0 is buy, 1 is sell
                # print("PICKED ACTION:", action)

            reward = trader.send((price, action))

            new_state = next(data_builder)

            if new_state is None:  # if last day in training set
                break

            if len(replay) < buffer:  # if buffer not filled, add to it
                replay.append((state, action, reward, new_state))
                state = new_state
            else:  # if buffer full, overwrite old values
                if h < (buffer - 1):
                    h += 1
                else:
                    h = 0
                replay[h] = (state, action, reward, new_state)
                # randomly sample our experience replay memory
                minibatch = random.sample(replay, batchSize)
                X_train = []
                y_train = []
                for memory in minibatch:
                    # Get max_Q(S',a)
                    old_state, action, reward, new_state = memory

                    old_qval = model.predict(
                        np.reshape(old_state["x"], (1, one_input_length)), batch_size=1
                    )
                    newQ = model.predict(
                        np.reshape(new_state["x"], (1, one_input_length)), batch_size=1
                    )
                    maxQ = np.max(newQ)
                    y = np.zeros((2,))
                    y[:] = old_qval[:]
                    update = reward + (gamma * maxQ)
                    # print("Before Update:", y)
                    # print("Update:", update)
                    if update == 0:  # no reward, and no updates in the future
                        status = 0
                        break

                    y[action] = update
                    # print("After Update:", y)
                    X_train.append(old_state["x"].reshape(one_input_length))
                    y_train.append(y.reshape(2))

                X_train = np.array(X_train)
                y_train = np.array(y_train)
                model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=0)
                state = new_state

    if epsilon > 0.1:
        epsilon -= 1 / epochs

model.save("reinforcement.h5")
