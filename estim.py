#!/usr/bin/env python
# coding:
###
# General Imports
import math, csv, random, time
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing

# Model Imports
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import chi2

#Evaluation Imports
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV

### Load data from csv file
def loadCsv(filename):
    lines = csv.reader(open(filename, newline= '', encoding='utf-8-sig'), delimiter=',', quotechar='|')
    dataset = []
    for row in lines:
        dataset.append([float(x) for x in row])
    return dataset

### Construct data frame
def data(filename='dummy.csv', scale=False):

    #Load data from file
    dataset = np.array(loadCsv(filename))

    n = len(dataset[0])
    X = np.array(dataset[:, :n-2])
    y = np.array(dataset[:, n-1])

    #Standardize and scale data
    if (scale):
        X = preprocessing.scale(X)

    return X, y

### Evaluate model
def evaluate(model_id, X, y, scale=False, seed=42):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    print("Fitting model on training set")
    t0 = time.time()
    grid = {'activation': ['logistic', 'tanh', 'relu'],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'hidden_layer_sizes': [(10, ), (20, ), (30, ), (40, ), (50, ), (10, 10, ), (20, 10, )], }
    clf = GridSearchCV(MLPRegressor(max_iter = 1000, random_state=seed), param_grid=grid, cv=5, iid=False, scoring='neg_mean_squared_error')
    clf.fit(X_train, y_train)
    print("done in %0.3fs" % (time.time() - t0))
    print("\nBest estimator found by grid search:")
    print('\t\t'+str(clf.best_estimator_))

    print("\nEvaluating best estimator on test set")
    t0 = time.time()
    y_pred = clf.predict(X_test)
    print("done in %0.3fs" % (time.time() - t0))

    score = round(mean_squared_error(y_test, y_pred), 4)
    print('\n\t\tRMSE (test):', score)

# ### Methods to run:
X, y = data()
evaluate(id, X, y, True, 42)
