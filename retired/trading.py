import math

"""
Trader class, takes price and signal and executes

Args:
	init_cash -- Initial cash
	ticker -- String identifying the stock

"""


class Trader:
    def __init__(self, init_cash=1000, ticker="STOCK"):
        self.init_cash = init_cash
        self.cash = self.init_cash
        self.ticker = ticker
        self.shares = 0
        self.open_position = False
        self.stats = {"buys": 0, "sells": 0}
        self._last_price = None

    def make_trade(self, prediction, price, shares=None):
        """ Makes a trade
			Args: prediction -- 0 for buy, 1 for sell
				  price -- price
				  shares -- number to buy (if buying). None = as many as possible
			Returns: self.cash
		"""
        self._last_price = price

        position_changed = False
        if prediction == 0:  # buy
            if not self.open_position:
                self._buy(price, shares)
                position_changed = True

        elif prediction == 1:  # sell
            if self.open_position:  # long position
                self._sell(price)
                position_changed = True

        return position_changed

    def _buy(self, price, shares):
        self.stats["buys"] += 1
        # print("Long on ticker", bought_ticker)
        if shares is None:
            self.shares = math.floor(self.cash / price)
        else:
            self.shares = shares

        val = self.shares * price
        self.cash -= val
        self.cash = round(self.cash, 2)
        print(
            "BUY:\tShares: %d\tprice: %.2f\tof ticker: %s\tvalue: %.2f"
            % (self.shares, price, self.ticker, val)
        )
        self.open_position = True

    def _sell(self, price):
        # print("Closed long on ticker", bought_ticker)
        self.stats["sells"] += 1
        net_value = round(self.shares * price, 2)
        print(
            "SELL:\tShares: %d\tprice: %.2f\tof ticker: %s\tvalue: %.2f"
            % (self.shares, price, self.ticker, net_value)
        )

        self.cash += net_value
        self.cash = round(self.cash, 2)
        print("Current cash:", self.cash)
        self.shares = 0
        self.open_position = False

    @property
    def total_value(self):
        return round(self.cash + self.shares * self._last_price, 2)

    def end_trading(self):
        """ Wraps up, return statistics and end results """
        if self.open_position:
            self.make_trade(1, self._last_price)

        self.stats["cash"] = self.cash
        return self.stats
