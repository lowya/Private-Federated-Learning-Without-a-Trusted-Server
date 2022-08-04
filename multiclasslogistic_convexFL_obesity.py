#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 23:12:29 2022

@author: Andrew Lowy

"""



import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
onehot_encoder = OneHotEncoder(sparse=False)
from numpy import mean
from numpy import median
from numpy import percentile
import pandas as pd
import matplotlib.pyplot as plt
import os
import sklearn
import statsmodels.api as sm
import statsmodels.formula.api as smf
import math
import itertools
from sklearn.model_selection import train_test_split

np.random.seed(1)
# ####LOAD and CLEAN THE DATA SET####
df = pd.read_csv('ObesityData.csv')



##Numerically Encode Categorical variables 
from sklearn.preprocessing import LabelEncoder
# #sex
le = LabelEncoder()
# #le.fit(df1.sex.drop_duplicates()) 
df['Gender'] = le.fit_transform(df['Gender'])
df['family_history_with_overweight'] = le.fit_transform(df['family_history_with_overweight'])
df['FAVC'] = le.fit_transform(df['FAVC'])
df['CAEC'] = le.fit_transform(df['CAEC'])
df['SMOKE'] = le.fit_transform(df['SMOKE'])
df['SCC'] = le.fit_transform(df['SCC'])
df['CALC'] = le.fit_transform(df['CALC'])
df['MTRANS'] = le.fit_transform(df['MTRANS'])
df['NObeyesdad'] = le.fit_transform(df['NObeyesdad'])
# df2['Month'] = le.fit_transform(df2['Month'])
# df1['Month'] = le.fit_transform(df1['Month'])
# df2['Month'] = le.fit_transform(df2['Month'])

#TRAIN/TEST SPLIT
(train, test) = sklearn.model_selection.train_test_split(df, test_size=0.2, train_size=0.8, random_state=None, shuffle=True, stratify=None)
X_train = train.iloc[:, :16]
Y_train = train.iloc[:, 16]
X_test = test.iloc[: , :16]
Y_test = test.iloc[:, 16]


###STANDARDIZE CONTINUOUS NUMERICAL FEATURES
standardized_Xtrain_cols = (X_train.iloc[::, [1,2,3,10, 13]] - X_train.iloc[::, [1,2,3,10, 13]].mean())/X_train.iloc[::, [1,2,3,10, 13]].std()
standardized_Xtest_cols = (X_test.iloc[::, [1,2,3,10, 13]] - X_train.iloc[::, [1,2,3,10, 13]].mean())/X_train.iloc[::, [1,2,3,10, 13]].std()
X_train = X_train.assign(Age = standardized_Xtrain_cols['Age'])
X_train = X_train.assign(Height = standardized_Xtrain_cols['Height'])
X_train = X_train.assign(Weight = standardized_Xtrain_cols['Weight'])
X_train = X_train.assign(CH2O = standardized_Xtrain_cols['CH2O'])
X_train = X_train.assign(TUE = standardized_Xtrain_cols['TUE'])
X_test = X_test.assign(Age = standardized_Xtest_cols['Age'])
X_test = X_test.assign(Height = standardized_Xtest_cols['Height'])
X_test = X_test.assign(Weight = standardized_Xtest_cols['Weight'])
X_test = X_test.assign(CH2O = standardized_Xtest_cols['CH2O'])
X_test = X_test.assign(TUE = standardized_Xtest_cols['TUE'])

train = pd.concat([X_train, Y_train], axis=1) #concatenate X_train and Y_train
####DIVIDE train data INTO N=7 CLIENTS based on obesity level
df0 = train[train["NObeyesdad"] == 0]
df1 = train[train["NObeyesdad"] == 1]
df2 = train[train["NObeyesdad"] == 2]
df3 = train[train["NObeyesdad"] == 3]
df4 = train[train["NObeyesdad"] == 4]
df5 = train[train["NObeyesdad"] == 5]
df6 = train[train["NObeyesdad"] == 6]
dfs = [df0, df1, df2, df3, df4, df5, df6]


def feat_lab_by_mach(dfs): #returns 2 lists (each contains N dataframes): features_by_machine, labels_by_machine
    features_by_machine = []  
    labels_by_machine = []
    for i in range(7):
        X = dfs[i].iloc[::, :16]
        X.insert(0, 'const', 1) #bias term
        Y = dfs[i].iloc[::, 16]
        features_by_machine.append(X)
        labels_by_machine.append(Y)
    return features_by_machine, labels_by_machine

feat_by_mach, lab_by_mach = feat_lab_by_mach(dfs)
X_test.insert(0, 'const', 1)

#Make client sets balanced (for simplicity of noise calibration):
client_sizes = [1,1,1,1,1,1,1]
for i in range(7):
    client_sizes[i] = len(lab_by_mach[i])


nmin = min(client_sizes)    

for i in range(7):
    rowstokeep = np.random.randint(0, feat_by_mach[i].shape[0], nmin)
    feat_by_mach[i], lab_by_mach[i] = feat_by_mach[i].iloc[rowstokeep, ::], lab_by_mach[i].iloc[rowstokeep]
    
X_train = pd.concat([feat_by_mach[0], feat_by_mach[1], feat_by_mach[2], feat_by_mach[3], feat_by_mach[4], feat_by_mach[5], feat_by_mach[6]]) #concatenate the 7 dfs in feat_by_mach into one (n*7) x 17 df 
Y_train = pd.concat([lab_by_mach[0], lab_by_mach[1], lab_by_mach[2], lab_by_mach[3], lab_by_mach[4], lab_by_mach[5], lab_by_mach[6]])

#Lip constant of softmax loss is bounded by 2J, where J = sup(||x_i||) (x_i is a feature vector for sample i)
J = 0 
for i in range(1477):
    nor = np.linalg.norm(X_train.iloc[i, ::])
    if nor > J:
        J = nor
#J = 9.214 => L <= 20
L = 20
######SOFTMAX REGRESSION######


def loss(X, Y, W):
    """
    Y: onehot encoded
    """
    Z = - X @ W
    N = X.shape[0]
    loss = 1/N * (np.trace(X @ W @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
    return loss

def gradient(W, X, Y):
    """
    Y: onehot encoded
    when using function for SGD, put X and Y to be matrices corresponding to minibatch of samples selected
    """
    Z = - X @ W
    P = softmax(Z, axis=1)
    #N = X.shape[0]
    gd = (X.T @ (Y - P))  
    return gd

def test_err(w, X_test, Y_test):  #computes prediction error given a parameter w and test data 
    errors = 0 #count number of errors 
    test_size = len(Y_test) #total number of test examples 
    Z = - X_test @ w
    P = softmax(Z, axis=1)
    for i in range(test_size): 
        prediction = np.argmax(P.iloc[i, ::])
        if prediction != Y_test[i]: 
            errors += 1
    return errors/test_size 

######FL ALGORITHMS########

def grad_eval(w, K, m): #stochastic (minibatch) grad eval of minibatch size K on client m; only used on train data 
            idxs = np.random.randint(0, len(lab_by_mach[m]), K) #draw minibatch (unif w/replacement) index set of size K from machine M's dataset  
            #Y = lab_by_mach[m].iloc[idxs]
            #Y_onehot = onehot_encoder.fit_transform(Y.reshape(-1,1))
            Y_onehot = np.zeros((K, 7))
            Y_onehot[::, m] = np.ones(K)   
            return gradient(w, feat_by_mach[m].iloc[idxs, :], Y_onehot) #returns average gradient across minibatch of size K


#MB-SGD
def minibatch_sgd(M, Mavail, K, R, stepsize):
    #dim = X[0].shape[1]
    Y_hot = np.zeros((7, n*7))
    for q in range(7): 
        for j in range(q*n, (q+1)*n):
            Y_hot[q,j] = 1  #Y_hot is one-hot encoding of Y_train; 7 x (n*7) matrix (each row is a std basis row vector)
    #losses = []
    #iterates = [np.zeros((17, 7))]
    w = np.zeros((17, 7))
    for r in range(R):
        g = np.zeros((17, 7)) #start with g = 0 vector 
        #randomly choose Mavail out of the M clients:
        S = np.random.choice(list(range(M)), size=Mavail, replace=False, p=None)
        for m in S:
            #w = iterates[-1]
            g += grad_eval(w, K, m)/Mavail #evaluate MB grad at last iterate  
        w = w - stepsize * g 
            #iterates.append(w - stepsize * g) #take MB-SGD step and add new iterate to list iterates 
        lossval = loss(X_train, Y_hot.T, w) #evaluate training loss on full training data set at last iterate
    return w, lossval  


#Local SGD 
def local_sgd_round(w_start, M, Mavail, K, stepsize):
    w_end = np.zeros_like(w_start) #initialize at 0
    #randomly choose Mavail out of the M clients:
    S = np.random.choice(list(range(M)), size=Mavail, replace=False, p=None)
    for m in S: #iterate through all Mavail available workers 
        w = w_start.copy()
        for _ in range(K): #K steps of local SGD 
            g = grad_eval(w, 1, m)
            w -= stepsize * g #one step 
        w_end += w / Mavail #average SGD updates across M clients  
    return w_end #return updated w

def local_sgd(M, Mavail, K, R, stepsize):
    Y_hot = np.zeros((7, n*7))
    for q in range(7): 
        for j in range(q*n, (q+1)*n):
            Y_hot[q,j] = 1  #Y_hot is one-hot encoding of Y_train; 7 x (n*7) matrix (each row is a std basis row vector)
    w = np.zeros((17, 7))
    #losses = []
    #iterates = [np.zeros(x_len)] #initialize list with x_len 0's 
    for r in range(R):
        w = local_sgd_round(w, M, Mavail, K, stepsize) #run local sgd round and add it to iterates
    lossval = loss(X_train, Y_hot.T, w)
    return w, lossval

#NOISY ALGS 
#For local SGD: add gauss(x_len, eps, delta, n, K*R, L, K) to gradient 

def gauss(shape, eps, delta, n, R, L): #moments account form of noise; shape should be a tuple, e.g. (17, 7)
    return np.random.normal(loc = 0, scale = np.sqrt((8*(L**2)*R*np.log(1/delta)/(n**2 * eps**2))), size = shape)

#Noisy MB-SGD: 
def dp_minibatch_sgd(M, Mavail, K, R, stepsize, eps, delta, L):
    #dim = X[0].shape[1]
    Y_hot = np.zeros((7, n*7))
    for q in range(7): 
        for j in range(q*n, (q+1)*n):
            Y_hot[q,j] = 1  #Y_hot is one-hot encoding of Y_train; 7 x (n*7) matrix (each row is a std basis row vector)
    #losses = []
    #iterates = [np.zeros((17, 7))]
    w = np.zeros((17, 7))
    for r in range(R):
        g = np.zeros((17, 7)) #start with g = 0 vector 
        #randomly choose Mavail out of the M clients:
        S = np.random.choice(list(range(M)), size=Mavail, replace=False, p=None)
        for m in S:
            #w = iterates[-1]
            g += grad_eval(w, K, m)/Mavail #evaluate MB grad at last iterate  
        w = w - stepsize * (g + gauss((17,7), eps, delta, n, R, L))
            #iterates.append(w - stepsize * g) #take MB-SGD step and add new iterate to list iterates 
        lossval = loss(X_train, Y_hot.T, w) #evaluate training loss on full training data set at last iterate
    return w, lossval  

#Noisy Local SGD:
def dp_local_sgd_round(w_start, M, Mavail, K, stepsize,  eps, delta, L):
    w_end = np.zeros_like(w_start) #initialize at 0
    #randomly choose Mavail out of the M clients:
    S = np.random.choice(list(range(M)), size=Mavail, replace=False, p=None)
    for m in S: #iterate through all Mavail available workers 
        w = w_start.copy()
        for _ in range(K): #K steps of local SGD 
            g = grad_eval(w, 1, m)
            w -= stepsize * (g + gauss((17,7), eps, delta, n, K*R, L)) #one step 
        w_end += w / Mavail #average SGD updates across M clients  
    return w_end #return updated w

def dp_local_sgd(M, Mavail, K, R, stepsize, eps, delta, L):
    Y_hot = np.zeros((7, n*7))
    for q in range(7): 
        for j in range(q*n, (q+1)*n):
            Y_hot[q,j] = 1  #Y_hot is one-hot encoding of Y_train; 7 x (n*7) matrix (each row is a std basis row vector)
    w = np.zeros((17, 7))
    #losses = []
    #iterates = [np.zeros(x_len)] #initialize list with x_len 0's 
    for r in range(R):
        w = dp_local_sgd_round(w, M, Mavail, K, stepsize, eps, delta, L) #run local sgd round and add it to iterates
    lossval = loss(X_train, Y_hot.T, w)
    return w, lossval

##################################################################################################################
p = 0 #for full heterogeneity
#dim = 100
dim = 17
M = 7
Mavail = 3
path = 'temp'
#n_m = load_MNIST2(p, dim, path)[-1] #number of examples (train and test) per digit per machine 
n = nmin
DO_COMPUTE = True

###User parameters - you can manually adjust these:### 
#num_trials = 1
#num_trials = 20 
#n_reps = 3 #number of runs per train split (for hyperparameter tuning)
n_reps = 1
n_stepsizes = 8
#n_stepsizes = 8

epsilons = [0.5, 1, 3, 6, 9]
delta = 1/(n**2)
#Note: With n = nmin = 202, we have 2*math.log(1/delta) = 21.23 (= maximal allowable epsilon for DP via moments account)
#R = 250
R = 100
K = int(max(1, n*math.sqrt(9/(4*R)))) #needed for privacy by moments account; 9 = largest epsilon that we test
mu = 0


######## TUNE STEPSIZES FOR EACH ALG 
mb_stepsizes = [np.exp(exponent) for exponent in np.linspace(-7,-1,n_stepsizes)]
loc_stepsizes = mb_stepsizes
#loc_stepsizes = [np.exp(exponent) for exponent in np.linspace(-9,-2,n_stepsizes)] 

#####Minibatch SGD######
print('Doing Minibatch SGD...') #for each stepsize option, compute train loss 
MB_results = np.zeros(n_stepsizes) #stores training loss values for each stepsize 
MB_iterates = [np.zeros((17,7)), np.zeros((17,7)), np.zeros((17,7)), np.zeros((17,7)), np.zeros((17,7)), np.zeros((17,7)), np.zeros((17,7)), np.zeros((17,7))]
#local_results = np.zeros(n_stepsizes)
        #MB_trains = np.zeros(len(gstepLproduct))
        #local_trains = np.zeros(len(cstepLproduct)) 
MB_tests = np.zeros(n_stepsizes)  #stores test errors for each stepsize 
#local_tests = np.zeros(n_stepsizes) 
for i, stepsize in enumerate(mb_stepsizes):
#w = np.zeros(dim)
    print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i+1, n_stepsizes))
    #w, l = minibatch_sgd(feat_by_mach, lab_by_mach, M, Mavail, K, R, stepsize)
    w, l = minibatch_sgd(M, Mavail, K, R, stepsize)
    MB_iterates[i] = w
    MB_results[i] = l #train loss 
    print('MB-SGD train loss for stepsize', stepsize, ':', l)
    #MB_tests[i] = test_err(w, X_test, Y_test)  

##STORE optimal test error##
opt_idx = np.argmin(MB_results)
w_opt = np.array(MB_iterates[opt_idx])
MB_test_error = test_err(w_opt, X_test, np.array(Y_test)) #copy and paste the  resulting test error into a separate file for storage. 
print('MB SGD test error is', MB_test_error)

##Local SGD#####
print('Doing Local SGD...') #for each stepsize option, compute train loss 
loc_results = np.zeros(n_stepsizes) #stores training loss values for each stepsize 
loc_iterates = [np.zeros((17,7)), np.zeros((17,7)), np.zeros((17,7)), np.zeros((17,7)), np.zeros((17,7)), np.zeros((17,7)), np.zeros((17,7)), np.zeros((17,7))]
#local_results = np.zeros(n_stepsizes)
        #MB_trains = np.zeros(len(gstepLproduct))
        #local_trains = np.zeros(len(cstepLproduct)) 
loc_tests = np.zeros(n_stepsizes)  #stores test errors for each stepsize 
#local_tests = np.zeros(n_stepsizes) 
for i, stepsize in enumerate(loc_stepsizes):
#w = np.zeros(dim)
    print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i+1, n_stepsizes))
    #w, l = minibatch_sgd(feat_by_mach, lab_by_mach, M, Mavail, K, R, stepsize)
    w, l = local_sgd(M, Mavail, K, R, stepsize)
    loc_iterates[i] = w
    loc_results[i] = l #train loss 
    print('Local SGD train loss for stepsize', stepsize, ':', l)
    #MB_tests[i] = test_err(w, X_test, Y_test)  

##STORE optimal test error##
opt_idx_loc = np.argmin(loc_results)
w_opt_loc = np.array(loc_iterates[opt_idx_loc])
loc_test_error = test_err(w_opt_loc, X_test, np.array(Y_test)) #copy and paste the  resulting test error into a separate file for storage. 
print('Local SGD test error is', loc_test_error)



###Noisy Algs####
epsilon = 3

###Noisy MB-SGD####
print('Doing DP Minibatch SGD...') #for each stepsize option, compute train loss 
DPMB_results = np.zeros(n_stepsizes) #stores training loss values for each stepsize 
DPMB_iterates = [np.zeros((17,7)), np.zeros((17,7)), np.zeros((17,7)), np.zeros((17,7)), np.zeros((17,7)), np.zeros((17,7)), np.zeros((17,7)), np.zeros((17,7))]
#local_results = np.zeros(n_stepsizes)
        #MB_trains = np.zeros(len(gstepLproduct))
        #local_trains = np.zeros(len(cstepLproduct)) 
DPMB_tests = np.zeros(n_stepsizes)  #stores test errors for each stepsize 
#local_tests = np.zeros(n_stepsizes) 
for i, stepsize in enumerate(mb_stepsizes):
#w = np.zeros(dim)
    print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i+1, n_stepsizes))
    #w, l = minibatch_sgd(feat_by_mach, lab_by_mach, M, Mavail, K, R, stepsize)
    w, l = dp_minibatch_sgd(M, Mavail, K, R, stepsize, epsilon, delta, L)
    DPMB_iterates[i] = w
    DPMB_results[i] = l #train loss 
    print('DP MB-SGD train loss for stepsize', stepsize, ':', l)
    #MB_tests[i] = test_err(w, X_test, Y_test)  

##STORE optimal test error##
opt_idx = np.argmin(DPMB_results)
w_opt = np.array(DPMB_iterates[opt_idx])
DPMB_test_error = test_err(w_opt, X_test, np.array(Y_test)) #copy and paste the  resulting test error into a separate file for storage. 
print('DP MB SGD test error is', DPMB_test_error)

###Noisy Local SGD####
dploc_stepsizes = [np.exp(exponent) for exponent in np.linspace(-9,-2,n_stepsizes)] #these smaller stepsizes work better for small (<=6) epsilon 
dploc_stepsizes2 = [np.exp(exponent) for exponent in np.linspace(-12,-2,n_stepsizes)] #even smaller (for eps <= 1)
print('Doing Noisy Local SGD...') #for each stepsize option, compute train loss 
dploc_results = np.zeros(n_stepsizes) #stores training loss values for each stepsize 
dploc_iterates = [np.zeros((17,7)), np.zeros((17,7)), np.zeros((17,7)), np.zeros((17,7)), np.zeros((17,7)), np.zeros((17,7)), np.zeros((17,7)), np.zeros((17,7))]
#local_results = np.zeros(n_stepsizes)
        #MB_trains = np.zeros(len(gstepLproduct))
        #local_trains = np.zeros(len(cstepLproduct)) 
dploc_tests = np.zeros(n_stepsizes)  #stores test errors for each stepsize 
#local_tests = np.zeros(n_stepsizes) 
#for i, stepsize in enumerate(loc_stepsizes):
for i, stepsize in enumerate(dploc_stepsizes2[0:5]):
#w = np.zeros(dim)
    print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i+1, n_stepsizes))
    #w, l = minibatch_sgd(feat_by_mach, lab_by_mach, M, Mavail, K, R, stepsize)
    w, l = dp_local_sgd(M, Mavail, K, R, stepsize, epsilon, delta, L)
    dploc_iterates[i] = w
    dploc_results[i] = l #train loss 
    print('DP Local SGD train loss for stepsize', stepsize, ':', l)
    #MB_tests[i] = test_err(w, X_test, Y_test)  

##STORE optimal test error##
opt_idx_dploc = np.argmin(dploc_results)
w_opt_dploc = np.array(dploc_iterates[opt_idx_dploc])
dploc_test_error = test_err(w_opt_dploc, X_test, np.array(Y_test)) #copy and paste the  resulting test error into a separate file for storage. 
print('DP Local SGD test error is', dploc_test_error)


