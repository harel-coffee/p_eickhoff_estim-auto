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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt

### Load data from csv file
def loadCsv(filename):
    lines = csv.reader(open(filename, newline= '', encoding='utf-8-sig'), delimiter=',', quotechar='|')
    dataset = []
    for row in lines:
        dataset.append([float(x) for x in row])
    return dataset

### Construct data frame
def data(filename='JFTO126.csv', scale=False):

    #Load data from file
    dataset = np.array(loadCsv(filename))
    
    n = len(dataset[0])
    for i in range(n):
        if dataset[i][1] == 0 or dataset[i][1] == 9:
            dataset = np.delete(dataset, i, 0)
    dataset = np.delete(dataset,0,1)
    dataset = np.delete(dataset,10,1)
    nn = len(dataset[0])
    X = np.array(dataset[:, :nn-1])
    y = np.array(dataset[:, nn-1])

    #Standardize and scale data
    if (scale):
        X = preprocessing.scale(X)
    
    
    return X, y

### Evaluate model
def evaluate(model_id, X, y, scale=False, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    print("Fitting model parameters on training set")
    t0 = time.time()
    grid = {'activation': ['identity','logistic', 'tanh', 'relu'], 'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'hidden_layer_sizes': [(10, ), (20, ), (30, ), (40, ), (50, ), (10, 10, ), (20, 10, )], }
    clf = GridSearchCV(MLPRegressor(max_iter = 1000, solver = 'lbfgs' , random_state=seed), param_grid=grid, cv=5, iid=False, scoring='neg_mean_squared_error')
    clf.fit(X_train, y_train)
    print("done in %0.3fs" % (time.time() - t0))
    print("\nBest estimator found by grid search:")
    print('\t'+str(clf.best_estimator_))

    print("\nEvaluating best estimator on test set")
    t0 = time.time()
    y_pred = clf.predict(X_test)
    print("done in %0.3fs" % (time.time() - t0))

    score = round(mean_absolute_error(y_test, y_pred), 4)
    print('\n\t\tMAE (test):', score)

    #Plot predicted vs. true twitch force per phase width
    widths = np.unique(X_test[:,1])
#    plt.figure(figsize=(10, 15))
#    plt.suptitle('Amplitude vs. Twitch Force (blue = true, red = predicted)')
#    for w in range(len(widths)):
#        p = plt.subplot(math.ceil(len(widths)/2), 2, w+1)
#        axes = plt.gca()
#        axes.set_ylim(0, 1.1)
#        for i in range (len(y_test)):
#            if X_test[i,1] == widths[w]:
#                plt.plot([X_test[i,0], X_test[i,0]], [y_pred[i], y_test[i]], color='black', linestyle='dashed', zorder=1)
#                plt.scatter(X_test[i,0], y_test[i], marker='o', color='blue', zorder=2)
#                plt.scatter(X_test[i,0], y_pred[i], marker='X', color='red', zorder=2)
#
#        p.title.set_text('(phase width = '+str(widths[w])+')')
#    plt.subplots_adjust(hspace=.8)
#    plt.savefig("plots.png")

# ### Methods to run:
X, y = data()
evaluate(id, X, y, True, 42)
#average through all files