#!/usr/bin/env python
# coding:
###
# General Imports
import math, csv, random, time
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
import glob

# Model Imports
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import chi2

#Evaluation Imports
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt

# Directory with CSV files
data_dir = "data"

### Load data from csv file
def loadCsv(filename):
	lines = csv.reader(open(filename, newline= '', encoding='utf-8-sig'), delimiter=',', quotechar='|')
	dataset = []
	for row in lines:
		if row[12] != "": # checking if the last cell in each row (column 12) is not empty
			dataset.append([float(x) for x in row])
	return dataset

### Construct data frame
def data(filename, scale):

	#Load data from file
	dataset = np.array(loadCsv(filename))
	indices_control_or_other = []

	# find rows that are Control (0) or Other (9)
	for i in range(len(dataset)):
		if dataset[i][1] == 0 or dataset[i][1] == 9:
			indices_control_or_other.append(i)

	dataset = np.delete(dataset, indices_control_or_other, 0) # delete rows that are Control/Other
	dataset = np.delete(dataset,[0, 11], 1) # deleting columns 0 and 11

	row_len = len(dataset[0])
	X = np.array(dataset[:, :row_len-1])
	y = np.array(dataset[:, row_len-1])
	# Cap ground truth within [0, 1]
	for i, truth in enumerate(y):
		if truth > 1:
			y[i] = 1
		if truth < 0:
			y[i] = 0
	#Standardize and scale data
	if (scale):
		X = preprocessing.scale(X)
	return X, y


### Evaluate model
def evaluate(model_id, X_train, y_train, X_test, y_test, X, y, seed=42):

	print("Fitting model parameters on training set")
	t0 = time.time()
	grid = {'activation': ['identity','logistic', 'tanh', 'relu'], 'learning_rate': ['constant', 'invscaling', 'adaptive'],
	'hidden_layer_sizes': [(10, ), (20, ), (30, ), (40, ), (50, ), (10, 10, ), (20, 10, )], }
	clf = GridSearchCV(MLPRegressor(max_iter = 1000, solver = 'adam' , random_state=seed), param_grid=grid, cv=5, iid=False, scoring='neg_mean_squared_error')
	clf.fit(X_train, y_train)
	print("done in %0.3fs" % (time.time() - t0))
	print("\nBest estimator found by grid search:")
	print('\t'+str(clf.best_estimator_))

	print("\nEvaluating best estimator on test set")

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

	index_cap = len(y_train)
	index_10 = np.random.randint(0,index_cap,10)
	index_20 = np.random.randint(0,index_cap,20)
	index_50 = np.random.randint(0,index_cap,50)
	index_100 = np.random.randint(0,index_cap,100)
	index_200 = np.random.randint(0,index_cap,200)

	all_index = [index_10,index_20,index_50,index_100,index_200]
	for i in all_index:
		length = len(i)
		print("Fitting model parameters on training set using", length, "rows of training set")
		t0 = time.time()
		grid = {'activation': ['identity','logistic', 'tanh', 'relu'], 'learning_rate': ['constant', 'invscaling', 'adaptive'],
		'hidden_layer_sizes': [(10, ), (20, ), (30, ), (40, ), (50, ), (10, 10, ), (20, 10, )], }
		new_clf = clf.best_estimator_
		#clf = GridSearchCV(MLPRegressor(max_iter = 1000, solver = 'lbfgs' , random_state=seed), param_grid=grid, cv=5, iid=False, scoring='neg_mean_squared_error')
		for j in range(1000):
			new_clf.partial_fit(X_train[i], y_train[i])
		print("done in %0.3fs" % (time.time() - t0))

		print("\nEvaluating best estimator on test set")
		t0 = time.time()
		y_pred = new_clf.predict(X_test)
		print("done in %0.3fs" % (time.time() - t0))

		score = round(mean_absolute_error(y_test, y_pred), 4)
		print('\n\t\tMAE (test):', score)





	# t0 = time.time()
	# y_pred = clf.predict(X_test)
	# print("done in %0.3fs" % (time.time() - t0))
	#
	# score = round(mean_absolute_error(y_test, y_pred), 4)
	# print('\n\t\tMAE (test):', score)
	#
	# scores.append(scores)

	#Plot predicted vs. true twitch force per phase width
	# widths = np.unique(X_test[:,1])
	# plt.figure(figsize=(10, 15))
	# plt.suptitle('Amplitude vs. Twitch Force (blue = true, red = predicted)')
	# for w in range(len(widths)):
	#     p = plt.subplot(math.ceil(len(widths)/2), 2, w+1)
	#     axes = plt.gca()
	#     axes.set_ylim(0, 1.1)
	#     for i in range (len(y_test)):
	#         if X_test[i,1] == widths[w]:
	#             plt.plot([X_test[i,0], X_test[i,0]], [y_pred[i], y_test[i]], color='black', linestyle='dashed', zorder=1)
	#             plt.scatter(X_test[i,0], y_test[i], marker='o', color='blue', zorder=2)
	#             plt.scatter(X_test[i,0], y_pred[i], marker='X', color='red', zorder=2)
	#     p.title.set_text('(phase width = '+str(widths[w])+')')
	# plt.subplots_adjust(hspace=.8)
	# plt.savefig("plots.png")

# ### Methods to run:

scores = []
test_files = [] #corresponding order with scores

for test_file in glob.iglob(data_dir + '/*.csv'):
	test_files.append(test_file)
	print(f"Starting training with held-out test file: {test_file}")
	train_X = None
	train_y = None
	test_X, test_y = data(test_file, True)
	for filepath in glob.iglob(data_dir + '/*.csv'):
		if filepath != test_file:

			file_X, file_y = data(filepath, True)
			if train_X is not None:
				train_X = np.concatenate((train_X, file_X))
				train_y = np.concatenate((train_y, file_y))
			else:
				train_X = file_X
				train_y = file_y
			print(f"Added {filepath} to training set")
	X, y = data(test_file, True)
	evaluate(id, train_X, train_y, test_X, test_y, X, y, 42)

for test_file, score in zip(test_files, scores):
	print(f"MAE {score} for held-out test subject {test_file}")



 # average through all files
