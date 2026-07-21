import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import minimize
from time import time

from visualizations import *

col_scheme_BR = ['midnightblue','indigo','mediumvioletred','maroon','darkorange','gold','yellow','olive','darkgreen','teal',"midnightblue","mediumblue","green","seagreen","firebrick","darkred","darkorange","burlywood","purple","magenta",]

plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 7

})


if __name__ == "__main__":
    ###############################
    #                             #
    #  Plotting generic figures   #
    #                             #
    ###############################
    if True:
        # plotCovSkSk(np.load("data/tmp/times_parallel_n80.npy"),80)
        plot_g2()
        plot_S()
        plot_transmission_only()
        plot_g_only()
        plot_dispersion([np.load("data/states/SS_n"+str(N)+".npy") for N in [20,40,100]])
        plot_cg_variance_EndMatter()
        plot_tilted_WTD(np.load("data/tmp/times_parallel_n20.npy"))


    ###############################
    #                             #
    # Plotting of Var T_n figures #
    #                             #
    ###############################
    if True:
        N_list = [10,20,40,80]
        times_list = []
        for N in N_list:
            times_list.append(np.load("data/tmp/times_parallel_n"+str(N)+".npy"))

        plot_cg_variance_parallel_LOGLOG(times_list,N_list)
        plot_cg_variance_parallel(times_list,N_list)

    ###############################
    #                             #
    # Appendix Var[Nt] comparison #
    #                             #
    ###############################
    if False:
        plot_varNT_comparisons(fresh=False)

    
    ###############################
    #                             #
    # Temperature stability plot  #
    #                             #
    ###############################
    if False:
        N = 20
        date_str = "20251008"

        s_list = np.load("data/times_finite/s_range_n"+str(N)+"_"+date_str+".npy")[2:-2]
        times_list = []

        for s in s_list:
            times_list.append(np.load("data/times_finite/times_parallel_n"+str(N)+"_s"+str(np.round(s,1))+".npy"))

        times_list.append(np.load("data/tmp/times_parallel_n"+str(N)+".npy"))

        s_list = np.concatenate((s_list,[np.inf]))
        plot_cg_variance_temps(times_list,s_list,N=20)