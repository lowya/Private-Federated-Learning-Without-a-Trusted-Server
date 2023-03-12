This repository contains code for the ICLR 2023 paper "Private Federated Learning Without a Trusted Server: Optimal Algorithms for Convex Losses," by Andrew Lowy &amp; Meisam Razaviyayn. The paper can be found at: https://openreview.net/forum?id=TVY6GoURrw

This paper studies federated learning (FL)—especially cross-silo FL—with data from people who do not trust the server or other silos. In this setting, each silo (e.g. hospital) has data from different people (e.g. patients) and must maintain the privacy of each person’s data (e.g. medical record), even if the server or other silos act as adversarial eavesdroppers. This requirement motivates the study of Inter-Silo Record-Level Differential Privacy (ISRL-DP), which requires silo i’s communications to satisfy record/item-level differential privacy (DP). ISRL-DP ensures that the data of each person (e.g. patient) in silo i (e.g. hospital i) cannot be leaked. ISRL-DP is different from well-studied privacy notions. Central and user-level DP assume that people trust the server/other silos. On the other end of the spectrum, local DP assumes that people do not trust anyone at all (even their own silo). Sitting between central and local DP, ISRL-DP makes the realistic assumption (in cross-silo FL) that people trust their own silo, but not the server or other silos. In this work, we provide tight (up to logarithms) upper and lower bounds for ISRL-DP FL with convex/strongly convex loss functions and homogeneous (i.i.d.) silo data. Remarkably, we show that similar bounds are attainable for smooth losses with arbitrary heterogeneous silo data distributions, via an accelerated ISRL-DP algorithm. We also provide tight upper and lower bounds for ISRL-DP federated empirical risk minimization, and use acceleration to attain the optimal bounds in fewer rounds of communication than the state-of-the-art. Finally, with a secure “shuffler” to anonymize silo messages (but without a trusted server), our algorithm attains the optimal central DP rates under more practical trust assumptions. Numerical experiments show favorable privacy-accuracy tradeoffs for our algorithm in classification and regression tasks.

![image](https://user-images.githubusercontent.com/59854605/224570763-657a6858-7b75-4eee-a32d-1fbd5e7a5769.png)


**Code Requirements:**
Our code requires Python 3 to run. 
Dependencies: math, numpy, matplotlib, torchvision, sklearn, pandas, scipy, itertools 

**Instructions for Reproducing the Plots in the Paper:** 
Feel free to change the user parameters (R, Mavail, epsilons, etc…) or select them as in the paper to reproduce the plots there. Note that N (defined in the paper) is denoted by M in the scripts, and M (defined in the papers) is denoted by Mavail. Once you have selected parameters, simply run the script (making sure you are in the proper directory) to reproduce the plots from the paper. You should create a folder called “data” in your current directory to store the mnist data when our script automatically downloads it for you. 

**Citation:**
If you find this repository useful in your research, please cite:
@article{lowy2021private,
  title={Private federated learning without a trusted server: Optimal algorithms for convex losses},
  author={Lowy, Andrew and Razaviyayn, Meisam},
  journal={arXiv preprint arXiv:2106.09779},
  year={2021}
}
