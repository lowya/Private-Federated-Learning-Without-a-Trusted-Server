#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 13:27:40 2021

@author: andrewlowy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: anonymous

"""

import math
import numpy as np
from numpy import mean
from numpy import median
from numpy import percentile
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
import pickle
import os
import itertools
import pandas as pd
from collections import defaultdict
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn import random_projection
from sklearn.model_selection import train_test_split


np.set_printoptions(precision=3, linewidth=240, suppress=True)
np.random.seed(2021)

q = 1/7 #fraction of mnist data we wish to use; q = 1 -> 8673 train examples per machine; q = 1/10 -> 867 train examples per machine
###Function to download and pre-process (normalize, PCA) mnist and store in "data" folder:
    ##Returns 4 arrays: train/test_features_by_machine = , train/test_labels_by_machine 
def load_MNIST2(p, dim, path):
    if path not in os.listdir('./data'):
        os.mkdir('./data/'+path)
    #if data folder is not there, make one and download/preprocess mnist: 
    if 'processed_mnist_features_{:d}.npy'.format(dim) not in os.listdir('./data/'+path):
        #convert image to tensor and normalize (mean 0, std dev 1):
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.,), (1.,)),])
        #download and store transformed dataset as variable mnist:
        mnist = datasets.MNIST('data', download=True, train=True, transform=transform)
        #separate features from labels and reshape features to be 1D array:
        features = np.array([np.array(mnist[i][0]).reshape(-1) for i in range(len(mnist))])
        labels = np.array([mnist[i][1] for i in range(len(mnist))])
        #apply PCA to features to reduce dimensionality to dim
        features = PCA(n_components=dim).fit_transform(features)
        #save processed features in "data" folder:
        np.save('data/' + path + '/processed_mnist_features_{:d}.npy'.format(dim), features)
        np.save('data/' + path + '/processed_mnist_labels_{:d}.npy'.format(dim), labels)
    #else (data is already there), load data:
    else:
        features = np.load('data/' + path + '/processed_mnist_features_{:d}.npy'.format(dim))
        labels = np.load('data/' + path + '/processed_mnist_labels_{:d}.npy'.format(dim))
    
    ## Group the data by digit
    #n_m = smallest number of occurences of any one digit (label) in mnist:
    #n_m = min([np.sum(labels == i) for i in range(10)]) #n_m = 5,421
    n_m = int(min([np.sum(labels == i) for i in range(10)])*q) #smaller scale version 
    #use defaultdict to avoid key error https://stackoverflow.com/questions/5900578/how-does-collections-defaultdict-work:
    by_number = defaultdict(list)
    #append feature vectors to by_number until there are n_m of each digit
    for i, feat in enumerate(features):
        if len(by_number[labels[i]]) < n_m:
            by_number[labels[i]].append(feat)
    #convert each list of n_m feature vectors (for each digit) in by_number to np array 
    for i in range(10):
        by_number[i] = np.array(by_number[i])

    ## Enumerate the even vs. odd tasks
    even_numbers = [0,2,4,6,8]
    odd_numbers = [1,3,5,7,9]
    #make list of all 25 pairs of (even, odd):
    even_odd_pairs = list(itertools.product(even_numbers, odd_numbers))

    ## Group data into 25 single even vs single odd tasks
    all_tasks = []
    for (e,o) in even_odd_pairs:
        #eo_feautres: concatenate even feats, odd feats for each e,o pair: 
        eo_features = np.concatenate([by_number[e], by_number[o]], axis=0)
        #(0,...,0, 1, ... ,1) labels of length 2*n_m corresponding to eo_features:
        eo_labels = np.concatenate([np.ones(n_m), np.zeros(n_m)])
        #concatenate eo_feautures and eo_labels into array of length 4*n_m: 
        eo_both = np.concatenate([eo_labels.reshape(-1,1), eo_features], axis=1)
        #add eo_both to all_tasks:
        all_tasks.append(eo_both)
    #all_tasks is a list of 25 ndarrays, each array corresponds to an (e,o) pair of digits (aka task) and is 10,842 (examples) x 101 (100=dim (features) plus 1 =dim(label))
    #all_evens: concatenated array of 5*n_m ones and 5*n_m = 27,105 feauture vectors (n_m for each even digit):
    all_evens = np.concatenate([np.ones((5*n_m,1)), np.concatenate([by_number[i] for i in even_numbers], axis=0)], axis=1)
    #all_odds: same thing but for odds and with zeros instead of ones:
    all_odds = np.concatenate([np.zeros((5*n_m,1)), np.concatenate([by_number[i] for i in odd_numbers], axis=0)], axis=1)
    #combine all_evens and _odds into all_nums (contains all 10*n_m = 54210 training examples):
    all_nums = np.concatenate([all_evens, all_odds], axis=0)

    ## Mix individual tasks with overall task
    #each worker m gets (1-p)* 2*n = (1-p)*10,842 examples from specific tasks and p*10,842 from mixture of all tasks. 
    #So p=1 -> homogeneous (zeta = 0); p=0 -> heterogeneous  
    features_by_machine = []  
    labels_by_machine = []
    n_individual = int(np.round(2*n_m * (1. - p))) #int (1-p)*2n_m = (1-p)*10,842
    n_all = 2*n_m - n_individual #=int p*2n_m  = p*10,842
    for m, task_m in enumerate(all_tasks): #m is btwn 0 and 24 inclusive
        task_m_idxs = np.random.choice(task_m.shape[0], size = n_individual) #specific: randomly choose (1-p)*2n_m examples from 2*n_m = 10,842 examples for task m (one (e,o) pair)
        all_nums_idxs = np.random.choice(all_nums.shape[0], size = n_all) #mixture of tasks: randomly choose p*2n_m examples from all 54,210 examples (all digits)
        data_for_m = np.concatenate([task_m[task_m_idxs, :], all_nums[all_nums_idxs, :]], axis=0) #machine m gets 10,842 total examples: fraction p are mixed, 1-p are specific to task m (one eo pair)
        features_by_machine.append(data_for_m[:,1:]) 
        labels_by_machine.append(data_for_m[:,0])
    features_by_machine = np.array(features_by_machine) #array of all 25 feauture sets (each set has 10,842 feauture vectors)
    labels_by_machine = np.array(labels_by_machine) #array of corresponding label sets
    ###Train/Test split for each machine###
    train_features_by_machine = []
    test_features_by_machine = []
    train_labels_by_machine = []
    test_labels_by_machine = []
    for m, task_m in enumerate(all_tasks):
        train_feat, test_feat, train_label, test_label = train_test_split(features_by_machine[m], labels_by_machine[m], test_size=0.20, random_state=1)
        train_features_by_machine.append(train_feat)
        test_features_by_machine.append(test_feat)
        train_labels_by_machine.append(train_label)
        test_labels_by_machine.append(test_label)
    train_features_by_machine = np.array(train_features_by_machine)
    test_features_by_machine = np.array(test_features_by_machine)
    train_labels_by_machine = np.array(train_labels_by_machine)
    test_labels_by_machine = np.array(test_labels_by_machine)
    print(train_features_by_machine.shape)
    return train_features_by_machine, train_labels_by_machine, test_features_by_machine, test_labels_by_machine, n_m

 


############################################## Logistic Regression ###############################################

def sigmoid(z):
    return 1. / (1. + np.exp(-np.clip(z, -15, 15))) #input is clipped i.e. projected onto [-15,15].  

# features ("X") is an [\widetilde{N} x d] matrix of features (each row is one data point x_i in R^d)
# labels is a \widetilde{N}-dimensional vector of labels (0/1)
#w is a d-dim vector of weights (parameters) 
def logistic_loss(w, features, labels): #returns average val of log loss over data = features, labels
    probs = sigmoid(np.dot(features,w))
    return (-1./features.shape[0]) * (np.dot(labels, np.log(1e-12 + probs)) + np.dot(1-labels, np.log(1e-12 + 1-probs))) #vectorized empirical loss with 1e-12 to avoid log(0)

def logistic_loss_gradient(w, features, labels):
    return np.dot(np.transpose(features), sigmoid(np.dot(features,w)) - labels) / features.shape[0] #dot here is used for matrix mult. result is d-vector

def logistic_loss_hessian(w, features, labels):
    s = sigmoid(np.dot(features, w))
    return np.dot(np.transpose(features) * s * (1 - s), features) / features.shape[0] #dot here is matrix mult: transpose(feat) * feat is dxd matrix  as desired
    
def test_err(w, features, labels):  #computes prediction error given a parameter w and data = features, labels
    errors = 0 #count number of errors 
    test_size = labels.shape[0]*labels.shape[1] #total number of test examples across all clients
    for i in range(features.shape[0]):
        for j in range(features.shape[1]):
            prob = sigmoid(np.dot(features[i,j], w))
            if prob > 0.5:
                prediction = 1
            else:
                prediction = 0
            if prediction != labels[i,j]:
                errors += 1
    return errors/test_size 

##################################################################################################################

def local_sgd_round(w_start, M, K, stepsize, grad_eval):
    w_end = np.zeros_like(w_start) #initialize at 0
    for m in range(M): #iterate through all M workers (assume M has been drawn for this round--they use fixed M=N; we'll implement random drawing of M_r, S_r)
        w = w_start.copy()
        for _ in range(K): #K steps of local SGD 
            g = grad_eval(w, 1, m)
            w -= stepsize * g #one step 
        w_end += w / M #average SGD updates across M clients  
    return w_end #return updated w

def noisy_local_sgd_round(w_start, M, K, stepsize, grad_eval):
    w_end = np.zeros_like(w_start) #initialize at 0
    for m in range(M): #iterate through all M workers (assume M has been drawn for this round or is fixed)
        w = w_start.copy()
        for _ in range(K): #K steps of local SGD 
            g = grad_eval(w, 1, m)
            w -= stepsize * (g + gauss_AC(x_len, eps, delta, n, K*R, L, K)) #one step 
        w_end += w / M #average SGD updates across M clients  
    return w_end #return updated w

def local_sgd(x_len, M, K, R, stepsize, loss_freq, f_eval, grad_eval, avg_window=8):
    losses = []
    iterates = [np.zeros(x_len)] #initialize list with x_len 0's 
    for r in range(R):
        if len(iterates) >= avg_window: 
            iterates = iterates[-(avg_window-1):] #just store the last avg_window-1 =7 iterates
        iterates.append(local_sgd_round(iterates[-1], M, K, stepsize, grad_eval)) #run local sgd round and add it to iterates
        if (r+1) % loss_freq == 0:
            losses.append(f_eval(np.average(iterates,axis=0))) #evalute f (at average of last 7 iterates) every loss_freq rounds 
            print('Iteration: {:d}/{:d}   Loss: {:f}                 \r'.format(r+1,R,losses[-1]), end='')
            if losses[-1] > 100:
                print('\nLoss is diverging: Loss = {:f}'.format(losses[-1]))
                return losses, 'diverged'
    print('')
    return iterates, losses, 'converged'


def minibatch_sgd(x_len, M, K, R, stepsize, loss_freq, f_eval, grad_eval, avg_window=8):
    losses = []
    iterates = [np.zeros(x_len)]
    for r in range(R):
        if len(iterates) >= avg_window:
            iterates = iterates[-(avg_window-1):] #just store the last avg_window-1 =7 iterates
        g = np.zeros(x_len) #start with g = 0 vector of dim x_len = 100 
        for m in range(M):
            g += grad_eval(iterates[-1], K, m) #evaluate stoch grad of log loss at last iterate 
        iterates.append(iterates[-1] - stepsize * g) #take SGD step and add new iterate to list iterates 
        if (r+1) % loss_freq == 0:
            losses.append(f_eval(np.average(iterates,axis=0))) #evalute f (at average of last 7 iterates) every loss_freq rounds and append to list "losses"
            print('Iteration: {:d}/{:d}   Loss: {:f}                 \r'.format(r+1,R,losses[-1]), end='')
            if losses[-1] > 100:
                print('\nLoss is diverging: Loss = {:f}'.format(losses[-1]))
                return iterates, losses, 'diverged'
    print('')
    return iterates, losses, 'converged'   

def ACnoisyMB_sgd(eps, delta, n, L, x_len, M, K, R, stepsize, loss_freq, f_eval, grad_eval, avg_window=8):
    losses = []
    iterates = [np.zeros(x_len)]
    for r in range(R):
        if len(iterates) >= avg_window:
            iterates = iterates[-(avg_window-1):] #just store the last avg_window-1 =7 iterates
        g = np.zeros(x_len) #start with g = 0 vector of dim x_len = 100 
        for m in range(M):
            g += grad_eval(iterates[-1], K, m) + gauss_AC(x_len, eps, delta, n, R, L, K) #evaluate stoch MB grad of log loss at last iterate, then add noise
        iterates.append(iterates[-1] - stepsize * g) #take SGD step and add new iterate to list iterates 
        if (r+1) % loss_freq == 0:
            losses.append(f_eval(np.average(iterates,axis=0))) #evalute f (at average of last 7 iterates) every loss_freq rounds and append to list "losses"
            print('Iteration: {:d}/{:d}   Loss: {:f}                 \r'.format(r+1,R,losses[-1]), end='')
            if losses[-1] > 100:
                print('\nLoss is diverging: Loss = {:f}'.format(losses[-1]))
                return losses, 'diverged'
    print('')
    return iterates, losses, 'converged' #returns log loss fxn value 

def ACnoisy_local_sgd(eps, delta, n, L, x_len, M, K, R, stepsize, loss_freq, f_eval, grad_eval, avg_window=8): #LDP (not CDP) variant of McMahon et al 2018
    losses = []
    iterates = [np.zeros(x_len)] #initialize list with x_len 0's 
    for r in range(R):
        if len(iterates) >= avg_window: 
            iterates = iterates[-(avg_window-1):] #just store the last avg_window-1 =7 iterates
        iterates.append(noisy_local_sgd_round(iterates[-1], M, K, stepsize, grad_eval)) #run local sgd round + noise, and add it to iterates
        if (r+1) % loss_freq == 0:
            losses.append(f_eval(np.average(iterates,axis=0))) #evalute f (at average of last 7 iterates) every loss_freq rounds 
            print('Iteration: {:d}/{:d}   Loss: {:f}                 \r'.format(r+1,R,losses[-1]), end='')
            if losses[-1] > 100:
                print('\nLoss is diverging: Loss = {:f}'.format(losses[-1]))
                return losses, 'diverged'
    print('')
    return iterates, losses, 'converged'

    
#Newton used to compute minimizer w^* and f^*
def newtons_method(w_len, f_eval, grad_eval, hessian_eval, max_iter=1000, tol=1e-6):
    w = np.zeros(w_len)
    stepsize = 0.5
    for t in range(max_iter):
        gradient = grad_eval(w)
        hessian = hessian_eval(w)
        update_direction = np.linalg.solve(hessian, gradient)
        w -= stepsize * update_direction
        newtons_decrement = np.sqrt(np.dot(gradient, update_direction))
        if newtons_decrement <= tol:
            print("Newton's method converged after {:d} iterations".format(t+1))
            return f_eval(w), w
    print("Warning: Newton's method failed to converge")
    return f_eval(w), w


def gauss_AC(d, eps, delta, n, R, L, K): #advanced composition form of noise
    return np.random.multivariate_normal(mean = np.zeros(d), cov = (256*(L**2)*R*(np.log(2.5*R*K/(delta*n))*np.log(2/delta))/(n**2 * eps**2))*np.eye(d))
#def gauss_AC(d, eps, delta, n, R, L, K): #moments account form of noise
    #return np.random.multivariate_normal(mean = np.zeros(d), cov = (8*(L**2)*R*np.log(1/delta)/(n**2 * eps**2))*np.eye(d))

##################################################################################################################
p = 0.0 #for full heterogeneity 
#dim = 100
dim = 50
M = 25
path = 'temp'
n_m = load_MNIST2(p, dim, path)[-1] #number of examples (train and test) per digit per machine 
n = int(n_m*2*0.8) #total number of TRAINING examples (two digits) per machine 
DO_COMPUTE = True

###User parameters - you can manually adjust these:### 
num_trials = 20  
#each trial involves a new train/test split for all N = M clients
loss_freq = 5 
#n_reps = 3 #number of runs per train split 
n_reps = 3
n_stepsizes = 10
#n_stepsizes = 8

epsilons = [0.75, 1.5, 3, 6, 12, 18]
delta = 1/(n**2)
#Rs = [15, 25]
#R = 15
R = 35 #then do R35&R50 with larger q = 1/4
#K = int(max(1, n*math.sqrt(18/(4*R)))) #needed for privacy by moments account; 18 = largest epsilon that we test
K = int(max(1, n*18/(4*math.sqrt(2*R*math.log(2/delta))))) #needed for privacy by advanced comp; 18 = largest epsilon that we test

path = 'dp_mnist_p={:.2f}_K={:d}_R={:d}'.format(p,K,R)

# avg_MB_test_error = 0
# avg_loc_test_error = 0

# avg_MB_train_error = 0
# avg_loc_train_error = 0

# noisyMB_test_errors = {}
# noisyMB_train_errors = {}
# noisyloc_test_errors = {}
# noisyloc_train_errors = {}


# local_ls = np.zeros(num_trials)
# MB_ls = np.zeros(num_trials)
# noisylocal_ls = {}
# noisyMB_ls = {}
# noisyMB_tests_trials = {}
# noisyloc_tests_trials = {}

local_ls = np.zeros(num_trials)
MB_ls = np.zeros(num_trials)
noisylocal_ls = {}
noisyMB_ls = {}
noisyMB_tests_trials = {}
noisyloc_tests_trials = {}


for eps in epsilons:
    noisylocal_ls[eps] = np.ones(num_trials)*1000
    noisyMB_ls[eps] = np.ones(num_trials)*1000
    noisyMB_tests_trials[eps] = np.ones(num_trials)*1000
    noisyloc_tests_trials[eps] = np.ones(num_trials)*1000

upsilons = np.zeros(num_trials)

MB_tests_trials = np.zeros(num_trials)
loc_tests_trials = np.zeros(num_trials)


if DO_COMPUTE:
    for trial in range(num_trials):
        print("DOING TRIAL", trial)
        train_features, train_labels, test_features, test_labels = load_MNIST2(p,dim,path)[0:4]
        x_len = train_features.shape[2] #dim of data (after PCA) = 100
        def f_eval(w):
            return logistic_loss(w, train_features.reshape(-1,x_len), train_labels.reshape(-1))

        def grad_eval(w, minibatch_size, m): #stochastic (minibatch) grad eval 
            idxs = np.random.randint(0,train_features[m].shape[0], minibatch_size) #draw minibatch (unif w/replacement) index set of size minibatch_size from machine m's dataset  
            return logistic_loss_gradient(w, train_features[m, idxs, :], train_labels[m, idxs]) #returns average gradient across minibatch of size minibatch_size (=K)

        def full_grad_eval(w):
            return logistic_loss_gradient(w, train_features.reshape(-1,x_len), train_labels.reshape(-1))

        def hessian_eval(w):
            return logistic_loss_hessian(w, train_features.reshape(-1,x_len), train_labels.reshape(-1))
        #Newton to compute Fstar, wstar, and zeta:
        Fstar, wstar = newtons_method(x_len, f_eval, full_grad_eval, hessian_eval) 
        for m in range(M):
            nrm_nabla_Fm_star = np.linalg.norm(grad_eval(wstar, len(train_labels[m]), m)) #norm of grad of F_m
            upsilons[trial] += nrm_nabla_Fm_star**2 / M
        print('Fstar = {:.6f}'.format(Fstar))
        print('zeta = {:.5f}'.format(upsilons[trial]))
        
        #compute Lipschitz constant of log loss: (Note L <= 2* max(np.linalg.norm(x)))
        l = np.zeros(train_features.shape[1])
        for i in range(train_features.shape[1]):
            l[i] = np.linalg.norm(train_features[1][i])
        mx = max(l)
        L = 2*mx
        lg_stepsizes = [np.exp(exponent) for exponent in np.linspace(-6,0,n_stepsizes)] #MB SGD
        lc_stepsizes = [np.exp(exponent) for exponent in np.linspace(-8,-1,n_stepsizes)] #Local SGD
        ###Non-private algorithms###
        print('Doing Minibatch SGD...') #for each stepsize option, compute average excess risk of MBSGD over n_reps trials
        MB_results = np.zeros(n_stepsizes)
        local_results = np.zeros(n_stepsizes)
        #MB_trains = np.zeros(len(gstepLproduct))
        #local_trains = np.zeros(len(cstepLproduct)) 
        MB_tests = np.zeros(n_stepsizes)
        local_tests = np.zeros(n_stepsizes) 
        for i, stepsize in enumerate(lg_stepsizes):
            #w = np.zeros(dim)
            print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i+1, n_stepsizes))
            for rep in range(n_reps):
                iterates, l, success = minibatch_sgd(x_len, M, K, R, stepsize, loss_freq, f_eval, grad_eval, avg_window=8)
                if success == 'converged':
                    MB_results[i] += (l[-1] - Fstar) / n_reps #average excess risk val over the n_reps= 4 trials 
                    MB_tests[i] += test_err(np.average(iterates, axis=0),test_features, test_labels)/n_reps 
                else:
                    MB_results[i] += 100
                    MB_tests[i] += 100
        MB_ls[trial] = np.min(MB_results) 
        MB_step_index = np.argmin(MB_results) 
        #noisyMB_w_opt = noisyMB_w[noisyMB_step_index]
        MB_tests_trials[trial] = MB_tests[MB_step_index] 
        print('Doing Local SGD...') #for each stepsize option, compute average excess risk of LocalSGD over 4 trials
        for i, stepsize in enumerate(lc_stepsizes):
            #w = np.zeros(dim)
            print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i+1, n_stepsizes))
            for rep in range(n_reps):
                iterates, l, success = local_sgd(x_len, M, K, R, stepsize, loss_freq, f_eval, grad_eval, avg_window=8)
                if success == 'converged':
                    local_results[i] += (l[-1] - Fstar) / n_reps #average excess risk over the n_reps= 4 trials
                    #w += np.average(iterates,axis=0) / n_reps 
                    local_tests[i] += test_err(np.average(iterates, axis=0), test_features, test_labels)/n_reps
                else:
                    local_results[i] += 100
                    local_tests[i] += 100
        local_ls[trial] = np.min(local_results) 
        local_step_index = np.argmin(local_results)  
        loc_tests_trials[trial] = local_tests[local_step_index]
        
        #######Noisy Algs#######
        noisyMB_trains = {}
        noisyMB_tests = {}
        noisyloc_trains = {}
        noisyloc_tests = {}
        for eps in epsilons:
            noisyMB_trains[eps] = np.zeros(n_stepsizes) 
            noisyMB_tests[eps]= np.zeros(n_stepsizes)
            noisyloc_trains[eps]=np.zeros(n_stepsizes)
            noisyloc_tests[eps]=np.zeros(n_stepsizes) 
        
        for eps in epsilons:
            print('\n\nDOING eps = {:f}'.format(eps))
            print('Doing Noisy MB SGD...') #for each stepsize option, compute average excess risk of MBSGD over 4 trials
            noisyMB_results = np.zeros(n_stepsizes) 
            noisyloc_results = np.zeros(n_stepsizes) 
            for i, stepsize in enumerate(lg_stepsizes): 
            #w = np.zeros(dim)
                print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i+1, n_stepsizes))
                for rep in range(n_reps): #n_reps=3 trials for each stepsize
                    iterates, l, success = ACnoisyMB_sgd(eps, delta, n, L, x_len, M, K, R, stepsize, loss_freq, f_eval, grad_eval, avg_window=8)
                    if success == 'converged':
                        noisyMB_results[i] += (l[-1] - Fstar) / n_reps 
                        noisyMB_tests[eps][i] += test_err(np.average(iterates, axis=0), test_features, test_labels)/n_reps
                    else:
                            noisyMB_results[i] += 100
                            noisyMB_tests[eps][i] += 100
            noisy_MB_l = np.min(noisyMB_results)  
            noisyMB_ls[eps][trial] = noisy_MB_l 
            noisyMB_step_index = np.argmin(noisyMB_results) 
            t = noisyMB_tests[eps][noisyMB_step_index] 
            print("noisy MB test error for trial {:d} is".format(trial), t)
            noisyMB_tests_trials[eps][trial] = t
        
            print('Doing Noisy Local GD...')  
            for i, stepsize in enumerate(lc_stepsizes): 
            #w = np.zeros(dim)
                print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i+1, len(lc_stepsizes)))
                for rep in range(n_reps):
                    iterates, l, success = ACnoisy_local_sgd(eps, delta, n, L, x_len, M, K, R, stepsize, loss_freq, f_eval, grad_eval, avg_window=8)
                    if success == 'converged':
                        noisyloc_results[i] += (l[-1] - Fstar) / n_reps  
                        noisyloc_tests[eps][i] += test_err(np.average(iterates, axis=0), test_features, test_labels)/n_reps
                    else:
                        noisyloc_results[i] += 100 
                        noisyloc_tests[eps][i] += 100 
            noisy_loc_l = np.min(noisyloc_results)
            noisylocal_ls[eps][trial] = noisy_loc_l 
            noisyloc_stepL_index = np.argmin(noisyloc_results)
            u = noisyloc_tests[eps][noisyloc_stepL_index] 
            print("noisy loc test error for trial {:d} is".format(trial), u)
            noisyloc_tests_trials[eps][trial] = u

      
print("noisy MB test errors", noisyMB_tests_trials)
print("noisy loc test errors", noisyloc_tests_trials)
print("MB test errors", MB_tests_trials)
print("local SGD test errors", loc_tests_trials)
print("upsilon^2", upsilons)

 #########PLOTS########
 
 ###error bar versions###
fig = plt.figure()
ax = fig.add_subplot(111)
#lower_p = 2.5
#upper_p = 97.5
lower_p = 5
upper_p = 95

#Noisy MB SGD#
noisyMB_errs = np.zeros(len(epsilons))
noisyMB_means = {}
noisyMB_lows = {}
noisyMB_his = {}
for e, eps in enumerate(epsilons):
    noisyMB_means[eps] = np.average(noisyMB_tests_trials[eps])
    noisyMB_lows[eps] = max(0.0, percentile(noisyMB_tests_trials[eps], lower_p))
    noisyMB_his[eps] = min(1.0, percentile(noisyMB_tests_trials[eps], upper_p))
    noisyMB_errs[e] = (noisyMB_his[eps] - noisyMB_lows[eps])/2

noisyMB_means_sorted = list(zip(*sorted(noisyMB_means.items())))[1] 
ax.errorbar(epsilons, noisyMB_means_sorted, yerr = noisyMB_errs, color = '#1f77b4', ecolor='lightblue', mfc='#1f77b4',
          mec='#1f77b4', capsize = 10, label='Noisy MB SGD after {:d} rounds'.format(R))

#Noisy Local SGD#
noisyloc_errs = np.zeros(len(epsilons))
noisyloc_means = {}
noisyloc_lows = {}
noisyloc_his = {}
for e, eps in enumerate(epsilons):
    noisyloc_means[eps] = np.average(noisyloc_tests_trials[eps])
    noisyloc_lows[eps] = max(0.0, percentile(noisyloc_tests_trials[eps], lower_p))
    noisyloc_his[eps] = min(1.0, percentile(noisyloc_tests_trials[eps], upper_p))
    noisyloc_errs[e] = (noisyloc_his[eps] - noisyloc_lows[eps])/2

noisyloc_means_sorted = list(zip(*sorted(noisyloc_means.items())))[1] 
ax.errorbar(epsilons, noisyloc_means_sorted, yerr = noisyloc_errs, color = '#ff7f0e', ecolor='navajowhite', mfc='#ff7f0e',
          mec='#ff7f0e',  capsize = 10, label='Noisy Local SGD after {:d} rounds'.format(R))


#MB SGD#
MB_errs = np.zeros(len(epsilons))
MB_mean = np.average(MB_tests_trials)
MB_low = max(0.0, percentile(MB_tests_trials, lower_p))
MB_hi = min(1.0, percentile(MB_tests_trials, upper_p))
for e, eps in enumerate(epsilons):
    MB_errs[e] = (MB_hi - MB_low)/2
ax.errorbar(epsilons, [MB_mean]*len(epsilons), yerr = MB_errs, color = '#2ca02c', ecolor='lightgreen', mfc='#2ca02c',
            mec='#2ca02c', capsize = 10,  label = 'MB SGD after {:d} rounds'.format(R))

#Local SGD#
loc_errs = np.zeros(len(epsilons))
loc_mean = np.average(loc_tests_trials)
loc_low = max(0.0, percentile(loc_tests_trials, lower_p))
loc_hi = min(1.0, percentile(loc_tests_trials, upper_p))
for e, eps in enumerate(epsilons):
    loc_errs[e] = (loc_hi - loc_low)/2
ax.errorbar(epsilons, [loc_mean]*len(epsilons), yerr = loc_errs, color = '#d62728', ecolor='lightcoral', mfc='#d62728',
          mec='#d62728',  capsize = 10, label = 'Local SGD after {:d} rounds'.format(R))


handles,labels = ax.get_legend_handles_labels()
ax.set_xlabel(r'$\epsilon$')
ax.set_ylabel('Avg. Test Error ({:d} Trials)'.format(num_trials))
#ax.set_title('K = {:d}, $\upsilon$={:.2f}, {:d} Trials'.format(K, np.average(upsilon), num_trials))  
ax.set_title(r'N = {:d}, K = {:d}, $\upsilon_*^2$={:.1f}'.format(M, K, np.average(upsilons)))
ax.legend(handles, labels, loc='upper right')
plt.savefig('plots' + path + 'errorbar_mnist_test_error_vs_epsilon.png', dpi=400)
plt.show()

###no error bars version###
###PLOT test error vs. epsilon###
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
noisyMB_test_errors_sorted = sorted(noisyMB_tests_trials.items()) # sorted by key, return a list of tuples
noisyloc_test_errors_sorted = sorted(noisyloc_tests_trials.items())
l = list(zip(*noisyMB_test_errors_sorted))[1]
m = []
for i in range(len(l)):
    m.append(np.average(l[i]))
l2 = list(zip(*noisyloc_test_errors_sorted))[1]
m2 = []
for i in range(len(l2)):
    m2.append(np.average(l2[i]))

#ax2.plot(epsilons, list(zip(*noisyMB_test_errors_sorted))[1], label='Noisy MB SGD after {:d} rounds'.format(R))
#ax2.plot(epsilons, list(zip(*noisyloc_test_errors_sorted))[1],label='Noisy Local SGD after {:d} rounds'.format(R))
ax2.plot(epsilons, m, label='Noisy MB SGD after {:d} rounds'.format(R))
ax2.plot(epsilons, m2,label='Noisy Local SGD after {:d} rounds'.format(R))
ax2.plot(epsilons, [np.average(MB_tests_trials)]*len(epsilons), label = 'MB SGD after {:d} rounds'.format(R))
ax2.plot(epsilons, [np.average(loc_tests_trials)]*len(epsilons), label = 'Local SGD after {:d} rounds'.format(R))
handles,labels = ax2.get_legend_handles_labels()
ax2.set_xlabel(r'$\epsilon$')
ax2.set_ylabel('Avg. Test Error ({:d} Trials)'.format(num_trials)) 
ax2.set_title(r'K = {:d}, $\upsilon_*^2$={:.2f}'.format(K, np.average(upsilons))) 
ax2.legend(handles, labels, loc='upper right')
plt.savefig('plots' + path + 'mnist_test_error_vs_epsilon.png', dpi=400)
plt.show()

  





