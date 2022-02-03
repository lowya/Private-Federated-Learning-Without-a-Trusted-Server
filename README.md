# Locally-Differentially-Private-Federated-Learning
Code for the paper "Private Federated Learning Without a Trusted Server: Optimal Algorithms for Convex Losses," by Andrew Lowy &amp; Meisam Razaviyayn. The paper can be found at: https://arxiv.org/abs/2106.09779

Our code requires Python 3 to run. 
Dependencies: math, numpy, matplotlib, torchvision, sklearn, pandas, scipy, itertools 

Feel free to change the user parameters (R, Mavail, epsilons, etc…) or select them as in the paper to reproduce the plots there. Note that N (defined in the paper) is denoted by M in the scripts, and M (defined in the papers) is denoted by Mavail. Once you have selected parameters, simply run the script (making sure you are in the proper directory) to reproduce the plots from the paper. You should create a folder called “data” in your current directory to store the mnist data when our script automatically downloads it for you. 
