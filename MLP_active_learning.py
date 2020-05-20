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
from modAL.models import ActiveLearner
from collections import defaultdict

# Model Imports
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import chi2

#Evaluation Imports
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt


#### USER PARAMETERS: ####
# Number of trials to run
num_trials = 20

# Number of rows for inital active learner training
num_init_rows = 5

# Directory with CSV files
data_dir = "data"

# Test ratio (for train-test split)
test_ratio = 0.2
#########################

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

# def MLP_std_query(regressor, X):
#     MLP = regressor.estimator
#     per_layer_weight = np.array([i for i in MLP.coefs_])
#     w_arr = np.std(per_layer_weight, axis=0)
#     query_idx = np.argmax(w_arr)
#     return query_idx, X[query_idx]


### Evaluate model
def evaluate(model_id, X, y, scale=False, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    

    # choose a small amount of random rows for initial training (e.g. 5)
    init_indices = np.random.choice(index_cap, num_init_rows, replace=False)
    print(f"Indices of intial train data: {init_indices}")
    init_X = X_train[init_indices]
    init_y = y_train[init_indices]

    # remove the initial rows from our train data
    X_train = np.delete(X_train, init_indices, axis=0)
    y_train = np.delete(y_train, init_indices, axis=0)

    
    for num_train in experiments:
        num_queries = num_train - num_init_rows

        t0 = time.time()
        learner = ActiveLearner(estimator=MLPRegressor(max_iter = 1000, solver = 'adam' , random_state=seed), query_strategy=uncertainty, X_training=init_X, y_training=init_y)

        for _ in range(num_queries):
            query_idx, query_instance = learner.query(X_train)
            learner.teach(X_train[query_idx].reshape(1, -1), np.array([y_train[query_idx]]))

        y_pred = learner.predict(X_test)
        print("done in %0.3fs" % (time.time() - t0))

        score = round(mean_absolute_error(y_test, y_pred), 4)
        
        # add score for this trial to results dictionary in correct experiment row
        results[num_train].append(score)

        print(f"MAE (test) with {num_train} rows: {score}")

def to_num(label):
    # remove all non-numeric characters
    label_str = ""
    for char in label:
        if char.isdigit():
            label_str += char
    return label_str

### Main Loop (Over Files)###
print(f"Active Learning with MLP Regressor.")
print(f"Model trained with {num_init_rows} rows before querying starts.")

for filepath in glob.iglob(data_dir + '/*.csv'):
    X, y = data(filepath, True)
    
    index_cap = int(len(X) * (1 - test_ratio))

    # num of training rows
    experiments = [10, 20, 50, 100, 200] 
    
    # add experiments for {500, 1000} if there are enough rows 
    if index_cap > 500:
        experiments.append(500)
    if index_cap > 1000:
        experiments.append(1000)

    # always run an experiment with all data
    experiments.append(index_cap)

    # results dictionary with "number of training rows" as keys
    results = defaultdict(list)

    # RUN TRIALS
    for trial in range(1, num_trials + 1):
        print(f"FILE {filepath}. TRIAL {trial}/{num_trials}.")
        evaluate(id, X, y, True, 42)

    # convert to DataFrame for csv saving
    results_df = pd.DataFrame.from_dict(results, orient="index")

    results_df.to_csv(f"MLP_ActiveLearn_{to_num(filepath)}.csv")


