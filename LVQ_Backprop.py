# This code helps to compare Backpropagation and Learning Vector Quantization algorithms
# Original Code for Backpropagation: https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
# Original Code for Learning Vector Quantization: https://machinelearningmastery.com/implement-learning-vector-quantization-scratch-python/
# The updated code is used as part of a homewrok assignment.
# Course : CIS 731 Artificial Neural Networks

# Backprop on the Seeds Dataset
from random import seed
from random import randrange
from random import random
from csv import reader
from math import sqrt
import csv
import math
import timeit
import numpy as np
from math import exp

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithms(dataset, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores_bkp = list()
	scores_lvq = list()
	lvq_count = 0
	bkp_count = 0
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		actual = [row[-1] for row in fold]
		t1_lvq = timeit.default_timer()
		predicted_lvq = learning_vector_quantization(train_set, test_set, *args)
		t2_lvq = timeit.default_timer()
		lvq_count = lvq_count + (t2_lvq-t1_lvq)
		accuracy_lvq = accuracy_metric(actual, predicted_lvq)
		scores_lvq.append(accuracy_lvq)
		t1_bkp = timeit.default_timer()
		predicted_bkp = back_propagation(train_set, test_set, *args)
		t2_bkp = timeit.default_timer()
		bkp_count = bkp_count + (t2_bkp-t1_bkp)
		accuracy_bkp = accuracy_metric(actual, predicted_bkp)
		scores_bkp.append(accuracy_bkp)
	writer.writerow([(sum(scores_bkp)/float(len(scores_bkp))),(sum(scores_lvq)/float(len(scores_lvq))),bkp_count,lvq_count,bkp_error_count,lvq_error_count])
	print('Mean Accuracy Bkp: %.3f%%' % (sum(scores_bkp)/float(len(scores_bkp))))
	print('Mean Accuracy lvq: %.3f%%' % (sum(scores_lvq)/float(len(scores_lvq))))

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected, res):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					global bkp_error_count
					bkp_error_count = bkp_error_count + 1
					error += (neuron['weights'][j] * neuron['delta']) 
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append((expected[j] - neuron['output']))
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate,expected):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j] #* cost[expected[j]][((-1)*int(math.log(neuron['output'])))]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		for row in train:
			outputs = forward_propagate(network, row)
			res = outputs.index(max(outputs))	
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			backward_propagate_error(network, expected, res)
			update_weights(network, row, l_rate, expected)

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	predictions = list()
	for row in test:
		prediction = predict(network, row)
		predictions.append(prediction)
	return(predictions)


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)
 
# Locate the best matching unit
def get_best_matching_unit(codebooks, test_row):
	distances = list()
	for codebook in codebooks:
		dist = euclidean_distance(codebook, test_row)
		distances.append((codebook, dist))
	distances.sort(key=lambda tup: tup[1])
	return distances[0][0]
    
# Make a prediction with codebook vectors
def predict_lvq(codebooks, test_row):
	bmu = get_best_matching_unit(codebooks, test_row)
	return bmu[-1]
 
# Create a random codebook vector
def random_codebook(train):
	n_records = len(train)
	n_features = len(train[0])
	codebook = [train[randrange(n_records)][i] for i in range(n_features)]
	return codebook
 
# Train a set of codebook vectors
def train_codebooks(train, lrate, epochs, n_codebooks):
	codebooks = [random_codebook(train) for i in range(n_codebooks)]
	for epoch in range(epochs):
		rate = lrate * (1.0-(epoch/float(epochs)))
		for row in train:
			bmu = get_best_matching_unit(codebooks, row)
			for i in range(len(row)-1):
				global lvq_error_count
				lvq_error_count = lvq_error_count + 1
				error = row[i] - bmu[i]
				if bmu[-1] == row[-1]:
					bmu[i] += rate * error
				else:
					bmu[i] -= rate * error
	return codebooks
 
# LVQ Algorithm
def learning_vector_quantization(train, test, lrate, epochs, n_codebooks):
	codebooks = train_codebooks(train, lrate, epochs, n_codebooks)
	predictions = list()
	for row in test:
		output = predict_lvq(codebooks, row)
		predictions.append(output)
	return(predictions)
    
# Test Backprop on Seeds dataset
seed(1)
# load and prepare data
filename = 'cardio_nsp.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)

#write accuracies, change file name based on the change in parameters
acc_filename = 'Comparison.csv'
oFile = open(acc_filename, "w")
writer = csv.writer(oFile, delimiter=',', dialect='excel', lineterminator='\n')

# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)

#modify below parameters and run the script
n_folds = 5
l_rate = 0.1
n_hidden = 10 #same as the number of codebooks
# evaluate algorithm
for n_epoch in range(100):
    bkp_error_count = 0
    lvq_error_count = 0
    evaluate_algorithms(dataset, n_folds, l_rate, n_epoch, n_hidden)