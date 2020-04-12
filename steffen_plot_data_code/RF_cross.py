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
test_data_dir = "representative_data"

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
    unprocessed_X = X
    y = np.array(dataset[:, row_len-1])

    #Standardize and scale data
    if (scale):
        X = preprocessing.scale(X)
    return X, y, unprocessed_X

### Evaluate model
def evaluate(model_id, X_train, y_train, X_test, y_test, seed=42):

    print("Fitting model parameters on training set")
    t0 = time.time()
    grid = {'n_estimators': (10, 50, 100, 1000),'min_samples_split': [2,5,10]}
    model = GridSearchCV(estimator = RandomForestRegressor(max_features = "auto", criterion = "mse", random_state = 42), param_grid=grid, cv=5, iid=False, scoring='neg_mean_squared_error')
    model.fit(X_train, y_train)
    print("done in %0.3fs" % (time.time() - t0))
    print("\nBest estimator found by grid search:")
    print('\t'+str(model.best_estimator_))

    print("\nEvaluating model on test set")
    t0 = time.time()
    y_pred = model.predict(X_test)
    
    print("done in %0.3fs" % (time.time() - t0))

    score = round(mean_absolute_error(y_test, y_pred), 4)
    print('\n\t\tMAE (test):', score)

    scores.append(score)

    return y_pred

# def plot(X_test, y_test, y_pred, label):
# #Plot predicted vs. true twitch force
#     first_phase_amp = np.abs(X_test[:,4])
#     second_phase_amp = np.abs(X_test[:,7])
#     amplitudes = first_phase_amp + second_phase_amp
#     plt.figure(figsize=(10, 15))
#     plt.suptitle('Amplitude vs. Twitch Force (blue = true, red = predicted)')
#     axes = plt.gca()
#     axes.set_ylim(0, 1.1)
#     plt.plot(amplitudes, y_test, 'bo')
#     plt.plot(amplitudes, y_pred, 'r+')

#     label_str = ""
#     for char in label:
#         if char.isalnum():
#             label_str += char

#     plt.savefig(f"plot_{label_str}.png")

def to_alnum(label):
    # remove all non-alphanumeric characters
    label_str = ""
    for char in label:
        if char.isalnum():
            label_str += char
    return label_str

def to_num(label):
    # remove all non-numeric characters
    label_str = ""
    for char in label:
        if char.digit():
            label_str += char
    return label_str



scores = []
test_files = [] #corresponding order with scores

for test_file in glob.iglob(test_data_dir + '/*.csv'):
    print(f"Starting training with held-out test file: {test_file}")
    X_train = None
    y_train = None
    X_test, y_test, X_test_unprocessed = data(test_file, True)
    for filepath in glob.iglob(data_dir + '/*.csv'):
        if to_num(filepath) != to_num(test_file):
            file_X, file_y, _ = data(filepath, True)
            if X_train is not None:
                X_train = np.concatenate((X_train, file_X))
                y_train = np.concatenate((y_train, file_y))
            else:
                X_train = file_X
                y_train = file_y
            print(f"Added {filepath} to training set")
    y_pred = evaluate(id, X_train, y_train, X_test, y_test, 42)

    # Save data for Steffen's plots
    data_for_plot = np.concatenate((X_test_unprocessed, y_test.reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1)
    np.savetxt(f"RF_cross_{to_num(test_file)}.csv", data_for_plot, delimiter=",")





for test_file, score in zip(test_files, scores):
    print(f"MAE {score} for held-out test subject {test_file}")

 # average through all files
