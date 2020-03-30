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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, RationalQuadratic, ExpSineSquared


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
  
    #Standardize and scale data
    if (scale):
        X = preprocessing.scale(X)
    return X, y

### Evaluate model
def evaluate(model_id, X_train, y_train, X_test, y_test, seed=42):

    print("Fitting model parameters on training set")
    t0 = time.time()

    model = GaussianProcessRegressor()
    
    model.fit(X_train, y_train)
    print("done in %0.3fs" % (time.time() - t0))

    print("\nEvaluating model on test set")
    t0 = time.time()
    y_pred = model.predict(X_test)
    print("done in %0.3fs" % (time.time() - t0))

    print(y_test)

    print(y_pred)

    score = round(mean_absolute_error(y_test, y_pred), 4)
    print('\n\t\tMAE (test):', score)

    scores.append(score)

    #Plot predicted vs. true twitch force per phase width
    widths = np.unique(X_test[:,1])
    plt.figure(figsize=(10, 15))
    plt.suptitle('Amplitude vs. Twitch Force (blue = true, red = predicted)')
    for w in range(len(widths)):
        p = plt.subplot(math.ceil(len(widths)/2), 2, w+1)
        axes = plt.gca()
        axes.set_ylim(0, 1.1)
        for i in range (len(y_test)):
            if X_test[i,1] == widths[w]:
                plt.plot([X_test[i,0], X_test[i,0]], [y_pred[i], y_test[i]], color='black', linestyle='dashed', zorder=1)
                plt.scatter(X_test[i,0], y_test[i], marker='o', color='blue', zorder=2)
                plt.scatter(X_test[i,0], y_pred[i], marker='X', color='red', zorder=2)
        p.title.set_text('(phase width = '+str(widths[w])+')')
    plt.subplots_adjust(hspace=.8)
    plt.savefig("plots.png")

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
    evaluate(id, train_X, train_y, test_X, test_y, 42)

for test_file, score in zip(test_files, scores):
    print(f"MAE {score} for held-out test subject {test_file}")

 # average through all files
