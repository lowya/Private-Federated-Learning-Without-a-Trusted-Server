Client-Level Local-Differentially-Private Federated-Learning

This repository contains code for the paper "Private Federated Learning Without a Trusted Server: Optimal Algorithms for Convex Losses," by Andrew Lowy &amp; Meisam Razaviyayn. The paper can be found at: https://arxiv.org/abs/2106.09779

Abstract: This paper studies federated learning (FL) in the absence of a trustworthy server/clients. In this
setting, each client needs to ensure the privacy of its own data, even if the server or other clients act
adversarially. This requirement motivates the study of local differential privacy (LDP) at the client level.
We provide tight (up to logarithms) upper and lower bounds for LDP convex/strongly convex federated
stochastic optimization with homogeneous (i.i.d.) client data. The LDP rates match the optimal statistical
rates in certain practical parameter regimes, resulting in “privacy for free.” Remarkably, we show that
similar rates are attainable for smooth losses with arbitrary heterogeneous client data distributions, via a
linear-time accelerated LDP algorithm. We also provide tight upper and lower bounds for LDP federated
empirical risk minimization (ERM). While tight upper bounds for ERM were provided in prior work, we
use acceleration to attain these bounds in fewer rounds of communication. Finally, with a secure “shuffler”
to anonymize client reports (but without the presence of a trusted server), our algorithm attains the
optimal central differentially private rates for stochastic convex/strongly convex optimization. Numerical
experiments validate our theory and show favorable privacy-accuracy tradeoffs for our algorithm.

Our code requires Python 3 to run. 
Dependencies: math, numpy, matplotlib, torchvision, sklearn, pandas, scipy, itertools 

Instructions for Reproducing the Plots in the Paper: 
Feel free to change the user parameters (R, Mavail, epsilons, etc…) or select them as in the paper to reproduce the plots there. Note that N (defined in the paper) is denoted by M in the scripts, and M (defined in the papers) is denoted by Mavail. Once you have selected parameters, simply run the script (making sure you are in the proper directory) to reproduce the plots from the paper. You should create a folder called “data” in your current directory to store the mnist data when our script automatically downloads it for you. 

Bibtex Citation: If you find this repository useful in your research, please cite:
@article{lowy2021private,
  title={Private federated learning without a trusted server: Optimal algorithms for convex losses},
  author={Lowy, Andrew and Razaviyayn, Meisam},
  journal={arXiv preprint arXiv:2106.09779},
  year={2021}
}
