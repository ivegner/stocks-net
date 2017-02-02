# Code is mostly not mine. See http://outlace.com/Reinforcement-Learning-Part-1/

import numpy as np
import random
import matplotlib.pyplot as plt

# random.seed(1)
# np.random.seed(1)

n = 10
arms = np.random.random(n)
epochs = 1000

av = np.ones(n) #initialize action-value array
counts = np.zeros(n) #stores counts of how many times we've taken a particular action

def reward(prob):
	total = 0;
	for i in range(10):
		if random.random() < prob:
			total += 1
	return total

#our bestArm function is much simpler now
def bestArm(a):
	return np.argmax(a) #returns index of element with greatest value

plt.xlabel("Plays")
plt.ylabel("Mean Reward")
for i in range(epochs):
	eps = n/i if i != 0 else 1
	if random.random() > eps:
		choice = bestArm(av)
		counts[choice] += 1
		k = counts[choice]
		rwd =  reward(arms[choice])
		old_avg = av[choice]
		new_avg = old_avg + (1/k)*(rwd - old_avg) #update running avg
		av[choice] = new_avg
	else:
		choice = np.where(arms == np.random.choice(arms))[0][0] #randomly choose an arm (returns index)
		counts[choice] += 1
		k = counts[choice]
		rwd =  reward(arms[choice])
		old_avg = av[choice]
		new_avg = old_avg + (1/k)*(rwd - old_avg) #update running avg
		av[choice] = new_avg

	#have to use np.average and supply the weights to get a weighted average
	runningMean = np.average(av, weights=np.array([counts[i]/np.sum(counts) for i in range(len(counts))]))
	plt.scatter(i, runningMean)
print(arms)
print(av)

plt.show()

