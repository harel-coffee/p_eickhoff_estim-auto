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
data_dir = "representative_data"

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
    # Cap ground truth within [0, 1]
    for i, truth in enumerate(y):
        if truth > 1:
            y[i] = 1
        if truth < 0:
            y[i] = 0
    #Standardize and scale data
    if (scale):
        X = preprocessing.scale(X)
    return X, y, unprocessed_X

### Evaluate model
def evaluate(model_id, X_train, y_train, X_test, y_test, seed=42):
    
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

    return y_pred


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


# ### Methods to run:
for filepath in glob.iglob(data_dir + '/*.csv'):
    X, y, unprocessed_X = data(filepath, True)
    np.random.seed(42)
    test_indices = np.random.choice(len(X), len(X)/0.2)
    X_train = np.delete(X, test_indices)
    y_train = np.delete(y, test_indices)
    X_test = X[test_indices]
    y_test = y[test_indices]

    X_test_unprocessed = unprocessed_X[test_indices]

    print("Evaluating:" + filepath)
    y_pred = evaluate(id, X_train, y_train, X_test, y_test, 42)

    # Save data for Steffen's plots
    data_for_plot = np.concatenate((X_test_unprocessed, y_test.reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1)
    np.savetxt(f"MLP_intra_{to_num(filepath)}.csv", data_for_plot, delimiter=",")


