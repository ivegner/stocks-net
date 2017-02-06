import matplotlib.pyplot as plt
_, plots = plt.subplots(2, sharex=True, squeeze = False)
all_x = [[1, 2, 3, 4], [1, 2, 3, 4]]
all_y = [[1, 2, 3, 4], [5, 6, 7, 8]]

for i in range(len(all_x)):
	x = all_x[i]
	y = all_y[i]
	plots[i][0].plot(x, y)

plt.show()