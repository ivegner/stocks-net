import numpy as np
import sys
import tensorflow as tf
from deap import base, creator, tools
from Stock_NN_Funcs import build_data
import random
# from mem_top import mem_top
import parmap

''' CONSTANTS '''
TEST_CASH = 10000
MARGIN_CASH = 10000


def neural_network_model(weights, biases, data):    
	layers = []
	for i, _ in enumerate(layer_sizes):
		if i != len(layer_sizes) - 1:    # If it's not the last element in layer_sizes (aka not the output size), give it weights and biases
			layers.append({"weights":tf.Variable(weights[i], name = "weights_"+str(i)),
						   "biases":tf.Variable(biases[i], name = "biases_"+str(i))})

			if i == 0:   # the first layer
				# print(data.get_shape(), layers[0]["weights"].get_shape(), layers[0]["biases"].get_shape())

				layers[0]["output"] = tf.add(tf.matmul(data, layers[0]["weights"]), layers[0]["biases"])
				layers[0]["output"] = tf.nn.sigmoid(layers[0]["output"])
			else:
				layers[i]["output"] = tf.add(tf.matmul(layers[i-1]["output"], layers[i]["weights"]), layers[i]["biases"])

				if i != len(layer_sizes) - 2:    # Apply sigmoid if it's not the last layer
					layers[i]["output"] = tf.nn.sigmoid(layers[i]["output"])

	return layers[-1]["output"]

def build_individual(container):
	weights, biases = [], []
	for i, size in enumerate(layer_sizes):
		if i != len(layer_sizes) - 1:    # If it's not the last element in layer_sizes (aka not the output size), give it weights and biases
			weights.append(np.random.normal(size = (size, layer_sizes[i+1],)).tolist())
			biases.append(np.random.normal(size = (layer_sizes[i+1])).tolist())
	
	# print("\n\nWEIGHTS:", weights, "\n")
	# print("BIASES:", biases, "\n\n")

	# weights = [float(weight) for layer in weights for weight in layer ]
	# biases = [float(bias) for layer in biases for bias in layer]

	return container([weights, biases])

def evaluate(individual):
	prediction = neural_network_model(individual[0], individual[1], x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# for epoch in range(hm_epochs):
		#   _, c = sess.run([optimizer, cost], feed_dict={x: train_data["X_norm"], y: train_data["Y"]})  #sets session placeholders to actual values

		cash = TEST_CASH
		shares = 0
		flag = 0
		short_price = 0
		output = tf.argmax(prediction, 1).eval({x:test_X}, session = sess)

		for day_price, bar in zip(prices, output):
			if bar == 2:    #buy
				# # print("buy")
				if flag == 0:       #no position
					shares = cash / day_price
					cash -= shares * day_price 
					flag = 1

				if flag == -1:    #short
					cash += shares * (short_price - day_price)
					shares = 0
					flag = 0

			elif bar == 0:    #sell
				# print("sell")
				if flag == 0:       # no position
					shares = MARGIN_CASH / day_price
					short_price = day_price
					flag = -1

				elif flag == 1:    # long
					cash += shares * day_price
					shares = 0
					flag = 0
	if flag == -1:
		cash += shares * (short_price - day_price)
	elif flag == 1:
		cash += shares * day_price

	return (cash,)

def mutate(individual, prob=0.5):
	for i in range(len(individual[0])):
		for j in range(len(individual[0][i])):
			for k in range(len(individual[0][i][j])):
				if random.random() < prob:
					individual[0][i][j][k] += random.gauss(0.0, 0.2)

	for i in range(len(individual[1])):
		for j in range(len(individual[1][i])):
			if random.random() < prob:
				individual[1][i][j] += random.gauss(0.0, 0.2)
	del individual.fitness.values

def mate(ind1, ind2):
	child1, child2 = [toolbox.clone(ind) for ind in (ind1, ind2)]
	assert len(child1) == len(child2)
	for i in range(len(child1[0])):
		tools.cxTwoPoint(child1[0][i], child2[0][i])
	for i in range(len(child1[1])):
		tools.cxTwoPoint(child1[1][i], child2[1][i])
	del child1.fitness.values
	del child2.fitness.values

N_IND = int(sys.argv[1])
N_GEN = int(sys.argv[2])
TEST_SEC = ["WIKI/MSFT", "WIKI/BAC"]

test_data = build_data(TEST_SEC)   
prices, test_X = test_data["price"], test_data["X_norm"]

# layer_sizes = [2, 5, 3]
layer_sizes = [len(test_X[0]), 1000, 1000, 1000, 1000, 1000, 1000, 3]   # the 3 is technically not a layer (it's the output), but it's here for convenience
x = tf.placeholder("float", [None, layer_sizes[0]])
y = tf.placeholder("float")

creator.create("FitnessMax", base.Fitness, weights = (1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", build_individual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", mate)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize = 3, k=N_IND)
toolbox.register("evaluate", evaluate)


print("Creating population...")
pop = toolbox.population(n=N_IND)
print("Evaluating initial population...")
fitnesses = parmap.map(toolbox.evaluate, pop)
for ind, fit in zip(pop, fitnesses):
	ind.fitness.values = fit

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

logbook = tools.Logbook()
logbook.header = "gen", "avg", "std", "min", "max"

print("And now we gon' evolve.")
for g in range(N_GEN):
	# print("Generation", g, "started training...")
	# print(mem_top())

	# num_parents = N_BEST + N_RANDOM
	# # Determine number of children based on total population and number of parents per generation
	# # total_num = num_parents + num_children * num_families -> N_IND = num_parents + num_children * num_parents/2 ->
	# # -> num_children = (N_IND - num_parents) / (num_parents/2) = 2(N_IND - num_parents) / num_parents
	# num_children = 2 * (N_IND - num_parents) / num_parents

	# # Select the next generation individuals
	# # print("Selecting individuals...")
	# parents = toolbox.selectBest(pop) + toolbox.selectRandom(pop)
	next_gen = toolbox.select(pop)

	# Clone the selected individuals (they're references before this) and shuffle
	next_gen = list(map(toolbox.clone, next_gen))
	random.shuffle(next_gen)

	# Apply crossover on the offspring
	# print("Mating... ( ͡° ͜ʖ ͡°)")
	for child1, child2 in zip(next_gen[::2], next_gen[1::2]):
		if random.random() < 0.5:
			toolbox.mate(child1, child2)
			del child1.fitness.values
			del child2.fitness.values

	# Apply mutation on the offspring
	# print("Mutating...")
	for mutant in next_gen:
		if random.random() < 0.3 and not mutant.fitness.valid:  #if it's not a parent
			toolbox.mutate(mutant)
			del mutant.fitness.values

	# Evaluate the individuals with an invalid fitness
	# print("Reevaluating fitness... RIP your CPU")
	invalid_ind = [ind for ind in next_gen if not ind.fitness.valid]
	fitnesses = parmap.map(toolbox.evaluate, invalid_ind)
	for ind, fit in zip(invalid_ind, fitnesses):
		ind.fitness.values = fit

	# The population is entirely replaced by the next gen
	pop[:] = next_gen
	record = stats.compile(pop)
	logbook.record(gen=g, **record)
	print(logbook.stream)

	
	print("Saving...")
	with tf.Session() as sess:
		best_ind = tools.selBest(pop, k=1)
		best_ind = toolbox.clone(best_ind)
		net = neural_network_model(best_ind[0][0], best_ind[0][1], x)
		sess.run(tf.global_variables_initializer())
		tf.train.Saver().save(sess, "./evolutionary.ckpt")


# Plotting!
gen = logbook.select("gen")
avg_fit = logbook.select("avg")
max_fit = logbook.select("max")
min_fit = logbook.select("min")

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
line1 = ax1.plot(gen, avg_fit, "b", label="Average Fitness")
line2 = ax1.plot(gen, max_fit, "r", label="Max Fitness")
line3 = ax1.plot(gen, min_fit, "g", label="Min Fitness")

ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness", color="k")
for tl in ax1.get_yticklabels():
	tl.set_color("k")

lns = line1 + line2 + line3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="bottom right")

plt.show()





