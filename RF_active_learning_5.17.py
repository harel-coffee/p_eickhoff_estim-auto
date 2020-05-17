# -*- coding: utf-8 -*-

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
from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling
from modAL.uncertainty import uncertainty_sampling


import warnings
warnings.filterwarnings("ignore" )

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
def evaluate(model_id, X, y, scale=False, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    index_cap = len(y_train)
    
    query_experiments = [10, 20, 50, 100, 200] 

    # add experiments for 500, 1000 if there are enough rows 
    if index_cap > 500:
        query_experiments.append(500)
    if index_cap > 1000:
        query_experiments.append(1000)

    query_experiments.append(index_cap)

    # choose 5 random 

    for n_queries in query_experiments:
        t0 = time.time()

        learner = ActiveLearner(estimator=RandomForestRegressor(max_features="auto", criterion="mse", n_estimators=100, random_state=42), query_strategy=uncertainty_sampling, X_training=X_train, y_training=y_train)
        
        
        for idx in range(n_queries):
            query_idx, query_instance = learner.query(X)
            
            learner.teach(X[query_idx].reshape(1, -1), y[query_idx].reshape(1, -1))


        y_pred = learner.predict(X_test)
        #print("done in %0.3fs" % (time.time() - t0))

        score = round(mean_absolute_error(y_test, y_pred), 4)
        #print('\n\t\tMAE (test):', score)
        iteration_scores.append(score)



    

    iteration_scores = []


        
    for i in all_index:
        length = len(i)
        #print("Fitting model parameters on training set using", length, "rows of training set" )
        #grid = {'n_estimators': (10, 50, 100, 1000),'min_samples_split': [2,5,10]}
        #clf = GridSearchCV(estimator = RandomForestRegressor(max_features = "auto", criterion = "mse", random_state = 42), param_grid=grid, cv=5, iid=False, scoring='neg_mean_squared_error')
        clf = ActiveLearner(estimator= RandomForestRegressor(max_features = "auto", criterion = "mse", random_state = 42), query_strategy=uncertainty_sampling, X_training = X_train[i], y_training=y_train[i])

        n_queries = 10
        for idx in range(n_queries):
            query_idx, query_instance = clf.query(X)
            clf.teach(X[query_idx].reshape(1, -1), y[query_idx].reshape(1, -1))

        #clf.fit(X_train[i], y_train[i])
        #print("done in %0.3fs" % (time.time() - t0))
        #print("\nBest estimator found by grid search:")
        #print('\t'+str(clf.best_estimator_))
    
        #print("\nEvaluating best estimator on test set")
        # t0 = time.time()
        y_pred = clf.predict(X_test)
        #print("done in %0.3fs" % (time.time() - t0))
    
        score = round(mean_absolute_error(y_test, y_pred), 4)
        #print('\n\t\tMAE (test):', score)
        iteration_scores.append(score)
    print(iteration_scores)
    


    #Plot predicted vs. true twitch force per phase width
#    widths = np.unique(X_test[:,1])
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
#        p.title.set_text('(phase width = '+str(widths[w])+')')
#    plt.subplots_adjust(hspace=.8)
#    plt.savefig("plots.png")

# ### Methods to run:

for filepath in glob.iglob(data_dir + '/*.csv'):
    X, y = data(filepath, True)
    print("Evaluating:" + filepath)
    for iteration in range(0, 20):
    	# print("Iteration:", iteration)
    	evaluate(id, X, y, True, 42)    
 # average through all files