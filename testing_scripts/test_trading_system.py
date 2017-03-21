import sys, os
# I welcome all suggestions for how to do this better
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))	 

import numpy as np
from Stock_NN_Funcs import build_data

price = [1, 2, 3, 4, 5]

prediction = [0, 1, 1, 1, 2]

CASH = 5
MARGIN_CASH = 5
shares = 0
flag = 0
short_price = 0

# print(output)
# print(price[:30])

for day_price, bar in zip(price, prediction):
	if bar == 2:    #buy
		# # print("buy")
		if flag == 0:       #no position
			# print("Long")
			shares = CASH / day_price
			print(shares, CASH)
			CASH -= shares * day_price 
			print(CASH)
			flag = 1

		if flag == -1:    #short
			# print("Closing short")
			CASH -= shares * day_price
			shares = 0
			flag = 0

	elif bar == 0:    #sell
		# print("sell")
		if flag == 0:       # no position
			# print("Short")
			shares = MARGIN_CASH / day_price
			CASH += shares * day_price
			flag = -1

		if flag == 1:    # long
			# print("Closing long")
			CASH += shares * day_price
			shares = 0
			flag = 0

if flag == -1:
	CASH -= shares * day_price
elif flag == 1:
	CASH += shares * day_price

	# saver.save(sess, "./"+PICKLE_NAME+".ckpt")

print("Returned $", int(CASH), " from an investment of $", 5)
