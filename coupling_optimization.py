import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import minimize
from time import time

from transmission_clock import noise_vs_signal_residue

col_scheme_BR = ['midnightblue','indigo','mediumvioletred','maroon','darkorange','gold','yellow','olive','darkgreen','teal',"midnightblue","mediumblue","green","seagreen","firebrick","darkred","darkorange","burlywood","purple","magenta",]

plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 7

})

def double_middle(x):
    """
    x   :   [x1,...,xn]

    returns [x1,...,xn/2,xn/2,xn/2+1,...,xn]
    """
    n = len(x)
    return np.concatenate((x[:int(n/2)],[x[int(n/2)]],x[int(n/2):]))

def optimize_sequentially(n_list,g_init=[],nsr_fun=noise_vs_signal_residue):
    """
    n_list = list of spin numbers to be optimized

    saves optimal guesses
    """
    # Vectorize
    n_list = np.atleast_1d(n_list)

    # Initial guess
    if not (len(g_init) == n_list[0]-1):
        g_init = np.ones((n_list[0]-1)) / 7
        g_init[0] = 0.33
        g_init[-1] = 0.33

    # Iterate through list
    for k,n in enumerate(n_list):
        print("Start optimizing n = "+str(n))
        if k==0:
            # Initial case
            res = minimize(nsr_fun,g_init,method="Nelder-Mead")
        else:
            # Load previous guess
            g_init = np.load("params/g_n"+str(n_list[k-1])+".npy")
            while len(g_init)<n-1:
                g_init = double_middle(g_init)
            res = minimize(nsr_fun,g_init,method="Nelder-Mead")
        
        # Save params
        np.save("params/g_n"+str(n)+".npy",res.x)
        print("Optimization complete\n")

    return None

def optimize_randomized(n_list,nsr_fun=noise_vs_signal_residue,n_random=1000):
    """
    n_list  :   list of spin numbers to be optimized
    nsr_fun :   function to be optimized
    """
    # Vectorize
    n_list = np.atleast_1d(n_list)

    # Fix U(1) symmetry
    fun = lambda x : np.log(nsr_fun(x**2))

    # Iterate through n_list
    for k,n in enumerate(n_list):
        print("Start optimizing n = "+str(n))

        res_x = []
        res_fun =  []

        # Iterate through randomized initial values
        for ell in range(n_random):
            g_init = np.random.random((n-1))
            res_ell = minimize(nsr_fun,g_init)
            
            res_x.append(np.sqrt(np.abs(res_ell.x)))
            res_fun.append(res_ell.fun)

            if ((ell+1) % 100) == 0:
                print(str(int((ell+1) / 10))+" % complete")
        
        # Find best
        res_fun = np.array(res_fun)
        g_opt = res_x[np.argmin(res_fun)]

        # Add Nelder-Mead
        res = minimize(nsr_fun,g_opt,method="Nelder-Mead")
        g_opt  = res.x

        # Save params
        np.save("params/g_n"+str(n)+".npy",g_opt)
        print("Optimization complete\n")

# Optimization
# n_list = np.arange(2,11)
# optimize_randomized(n_list)

n_list = np.arange(10,101)
optimize_sequentially(n_list,g_init=np.load("params/g_n"+str(min(n_list))+".npy"),nsr_fun=noise_vs_signal_residue)