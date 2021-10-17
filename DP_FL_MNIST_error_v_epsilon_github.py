#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrew Lowy

"""
import math
import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
import pickle
import os
import itertools
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
num_trials = 2 #due to time/computation constraints, we recommend keeping num_trials = 1 or 2
#each trial involves a new train/test split for all N = M clients
loss_freq = 5 
n_reps = 3 #number of runs per train split 
n_stepsizes = 10 
#n_stepsizes = 8

epsilons = [0.75, 1.5, 3, 6, 12, 18]
delta = 1/(n**2)
#Rs = [15, 25]
#R = 15
R = 50 #then do R35&R50 with larger q = 1/4
#K = int(max(1, n*math.sqrt(18/(4*R)))) #needed for privacy by moments account; 18 = largest epsilon that we test
K = int(max(1, n*18/(4*math.sqrt(2*R*math.log(2/delta))))) #needed for privacy by advanced comp; 18 = largest epsilon that we test


path = 'dp_mnist_p={:.2f}_K={:d}_R={:d}'.format(p,K,R)

avg_MB_test_error = 0
avg_loc_test_error = 0

avg_MB_train_error = 0
avg_loc_train_error = 0

noisyMB_test_errors = {}
noisyMB_train_errors = {}
noisyloc_test_errors = {}
noisyloc_train_errors = {}


#keep track of train excess risk too
local_ls = 0
MB_ls = 0
noisylocal_ls = {}
noisyMB_ls = {}

for eps in epsilons:
    noisyMB_test_errors[eps] = 0
    noisyMB_train_errors[eps] = 0
    noisyloc_test_errors[eps] = 0
    noisyloc_train_errors[eps] = 0
    noisylocal_ls[eps] = 0
    noisyMB_ls[eps] = 0
    
zeta = np.zeros(num_trials)
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
            zeta[trial] += nrm_nabla_Fm_star**2 / M
        print('Fstar = {:.6f}'.format(Fstar))
        print('zeta = {:.5f}'.format(zeta[trial]))
        
        #compute Lipschitz constant of log loss: (Note L <= 2* max(np.linalg.norm(x)))
        l = np.zeros(train_features.shape[1])
        for i in range(train_features.shape[1]):
            l[i] = np.linalg.norm(train_features[1][i])
        mx = max(l)
        L = 2*mx
        lg_stepsizes = [np.exp(exponent) for exponent in np.linspace(-6,0,n_stepsizes)] #MB SGD
        lc_stepsizes = [np.exp(exponent) for exponent in np.linspace(-8,-1,n_stepsizes)] #Local SGD
        print('Doing Minibatch SGD...') #for each stepsize option, compute average excess risk of MBSGD over n_reps trials
        MB_results = np.zeros((R//loss_freq, len(lg_stepsizes)))
        local_results = np.zeros((R//loss_freq, len(lc_stepsizes)))
        MB_w = [np.zeros(dim)]*len(lg_stepsizes)
        local_w = [np.zeros(dim)]*len(lc_stepsizes)
        for i,stepsize in enumerate(lg_stepsizes):
            print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i+1, len(lg_stepsizes)))
            for rep in range(n_reps):
                iterates, l, success = minibatch_sgd(x_len, M, K, R, stepsize, loss_freq, f_eval, grad_eval)
                if success == 'converged':
                    MB_results[:,i] += (l - Fstar) / n_reps #store in ith col: average excess risk vals (for all R//loss_freq iterates) over the n_reps= 4 trials 
                    MB_w[i] += sum(iterates) / (len(iterates)*n_reps) # average of last avg_window iterates 
                else:
                    MB_results[:,i] += 100
        print('Doing Local SGD...') #for each stepsize option, compute average excess risk of LocalSGD over nreps trials
        for i,stepsize in enumerate(lc_stepsizes):
            print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i+1, len(lc_stepsizes)))
            for rep in range(n_reps):
                iterates, l, success = local_sgd(x_len, M, K, R, stepsize, loss_freq, f_eval, grad_eval)
                if success == 'converged':
                    local_results[:,i] += (l - Fstar) / n_reps #average excess risk over the n_reps trials
                    local_w[i] += sum(iterates) / (len(iterates)*n_reps) 
                else:
                    local_results[:,i] += 100
        local_l = np.min(local_results, axis = 1)
        local_step_index = np.argmin(local_results, axis = 1) #1d array (of len = R//loss_freq)indices of optimal stepsizes at iter5, ..., iterR
        final_local_step_index = local_step_index[-1] #stepsize that minimizes loss at final iteration
        MB_l = np.min(MB_results, axis=1)
        MB_step_index = np.argmin(MB_results, axis = 1) 
        final_MB_step_index = MB_step_index[-1]
        local_test_error = test_err(local_w[final_local_step_index],test_features, test_labels) 
        MB_test_error = test_err(MB_w[final_MB_step_index], test_features, test_labels)
        local_train_error = test_err(local_w[final_local_step_index],train_features, train_labels)
        MB_train_error = test_err(MB_w[final_MB_step_index], train_features, train_labels)
        
        avg_MB_test_error += MB_test_error/num_trials
        avg_loc_test_error += local_test_error/num_trials
        avg_MB_train_error += MB_train_error/num_trials
        avg_loc_train_error += local_train_error/num_trials
        
        local_ls += local_l[-1]/num_trials
        MB_ls += MB_l[-1]/num_trials
        
    
    ####Noisy algorithms####
        for eps in epsilons:
            print('\n\nDOING eps = {:f}'.format(eps))
            print('Doing Noisy MB SGD...') #for each stepsize option, compute average excess risk of MBSGD over n_reps trials
            noisyMB_results = np.zeros((R//loss_freq, len(lg_stepsizes))) #store (e.g. 10 x 10) loss results for R//loss_freq (e.g. = 10) iterates for each (e.g. of 10) stepsize
            noisyMB_w = [np.zeros(dim)]*len(lg_stepsizes)
            for i,stepsize in enumerate(lg_stepsizes):
                print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i+1, len(lg_stepsizes)))
                for rep in range(n_reps): #n_reps=4 trials for each stepsize
                    iterates, l, success = ACnoisyMB_sgd(eps, delta, n, L, x_len, M, K, R, stepsize, loss_freq, f_eval, grad_eval)
                    if success == 'converged':
                        noisyMB_results[:,i] += (l - Fstar) / n_reps #each rep we add average excess risk over the n_reps runs to NoisyMB_results[:,i]
                        #after all n_reps reps, NoisyMB_results[:,i] = l - Fstar = excess loss for i-th stepsize 
                        noisyMB_w[i] += sum(iterates) / (len(iterates)*n_reps) #return average w over reps and last avg_window -1 = 7 iterations 
                    else:
                        noisyMB_results[:,i] += 100        
            noisy_MB_l = np.min(noisyMB_results, axis=1) #array of minimum (over all stepsizes) loss value at each of 10 spaced out iterates 
            noisyMB_step_index = np.argmin(noisyMB_results, axis = 1) 
            final_noisyMB_step_index = noisyMB_step_index[-1]
            print("optimal noisy MB stepsize for eps = {:f} is".format(eps), lg_stepsizes[final_noisyMB_step_index])
            noisyMB_test_errors[eps] += test_err(noisyMB_w[final_noisyMB_step_index], test_features, test_labels)/num_trials
            noisyMB_train_errors[eps] += test_err(noisyMB_w[final_noisyMB_step_index], train_features, train_labels)/num_trials
        
            print('Doing Noisy Local GD...') #for each stepsize option, compute average excess risk of MBSGD over nreps runs
            noisyloc_results = np.zeros((R//loss_freq, len(lc_stepsizes))) #store (e.g. 10 x 10) loss results for R//loss_freq (e.g. = 10) iterates for each (e.g. of 10) stepsize
            noisyloc_w = [np.zeros(dim)]*len(lc_stepsizes)
            for i,stepsize in enumerate(lc_stepsizes):
                print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i+1, len(lc_stepsizes)))
                for rep in range(n_reps): #n_reps=4 trials for each stepsize
                    iterates, l, success = ACnoisy_local_sgd(eps, delta, n, L, x_len, M, K, R, stepsize, loss_freq, f_eval, grad_eval)
                    if success == 'converged':
                        noisyloc_results[:,i] += (l - Fstar) / n_reps #each rep we add average excess risk over the n_reps trials to NoisyMB_results[:,i]
                #after all n_reps reps, NoisyMB_results[:,i] = l - Fstar = excess loss for i-th stepsize 
                        noisyloc_w[i] += sum(iterates) / (len(iterates)*n_reps)
                    else:
                        noisyloc_results[:,i] += 100    
            noisy_loc_l = np.min(noisyloc_results, axis = 1)
            noisyloc_step_index = np.argmin(noisyloc_results, axis = 1)
            final_noisyloc_step_index = noisyloc_step_index[-1]
            print("optimal noisy loc stepsize for eps = {:f} is".format(eps), lc_stepsizes[final_noisyloc_step_index])
            noisyloc_test_errors[eps] += test_err(noisyloc_w[final_noisyloc_step_index], test_features, test_labels)/num_trials
            noisyloc_train_errors[eps] += test_err(noisyloc_w[final_noisyloc_step_index], train_features, train_labels)/num_trials
            
            noisylocal_ls[eps] += noisy_loc_l[-1]/num_trials
            noisyMB_ls[eps] += noisy_MB_l[-1]/num_trials
        

print("noisy MB test errors", noisyMB_test_errors)
print("noisy loc test errors", noisyloc_test_errors)
print("MB test error", avg_MB_test_error)
print("local SGD test error", avg_loc_test_error)

###PLOT test error vs. epsilon###
fig = plt.figure()
ax = fig.add_subplot(111)
noisyMB_test_errors_sorted = sorted(noisyMB_test_errors.items()) # sorted by key, return a list of tuples
noisyloc_test_errors_sorted = sorted(noisyloc_test_errors.items())
ax.plot(epsilons, list(zip(*noisyMB_test_errors_sorted))[1], label='Noisy MB SGD after {:d} rounds'.format(R))
ax.plot(epsilons, list(zip(*noisyloc_test_errors_sorted))[1],label='Noisy Local SGD after {:d} rounds'.format(R))
ax.plot(epsilons, [avg_MB_test_error]*len(epsilons), label = 'MB SGD after {:d} rounds'.format(R))
ax.plot(epsilons, [avg_loc_test_error]*len(epsilons), label = 'Local SGD after {:d} rounds'.format(R))
handles,labels = ax.get_legend_handles_labels()
ax.set_xlabel(r'$\epsilon$')
ax.set_ylabel('Test Error') 
ax.set_title(r'K = {:d}, $\upsilon_*^2$={:.2f}'.format(K, np.average(zeta))) 
ax.legend(handles, labels, loc='upper right')
plt.savefig('plots' + path + 'test_error_vs_epsilon.png', dpi=400)
plt.show()











