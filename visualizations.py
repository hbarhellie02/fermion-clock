import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.optimize import minimize


from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from scipy.linalg import eigh
from scipy.special import j0, j1, sici
from scipy.integrate import cumulative_trapezoid

from transmission_clock import g2, bulk_current_COV_finite, noise_vs_signal_COV_finite_time, noise_vs_signal_residue, noise_vs_signal_COV_FW_ONLY, noise_vs_signal_COV, T

col_scheme_BR = ['midnightblue','indigo','mediumvioletred','maroon','darkorange','gold','yellow','olive','darkgreen','teal',"midnightblue","mediumblue","green","seagreen","firebrick","darkred","darkorange","burlywood","purple","magenta",]

plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 7

})

def plot_S():
    """
    plotting of noise
    """
    ## PART C / Noise
    N_full = (np.rint(10**np.linspace(0.8,2,30))).astype(int) #np.arange(5,101)
    S_range = np.zeros((len(N_full)))

    for kN, N in enumerate(N_full):
        # Comment for later
        # check whether I should renormalize g or whether I should
        # simply look at S/I ... both ok.
        g = np.load("params/g_n"+str(N)+".npy")
        # g = g # * (g_ref / g[int(n/2)])

        S, I = noise_vs_signal_residue(g,True)
        S_range[kN] = S/I

    plt.figure(figsize=(2.3,1.3))

    z = np.polyfit(np.log(N_full),np.log(S_range),deg=1)
    plt.loglog(N_full,np.exp(z[1])*N_full**z[0],
                label=r"fit $S\sim N^{"+str(np.round(z[0],1))+"}$")
    plt.loglog(N_full,S_range,linestyle="",marker=".",
                label="data",color="black")
    
    print("Scaling exponent: "+str(np.round(z[0],3)))

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.5)  # or 0 for no spacing
    plt.savefig("figs/DJ.svg")
    plt.show()

def plot_g_only():
    """
    Plotting of couplings function for Fig 1
    """
    # Which n to plot
    n_range = np.array([40])
    g_list = []

    for n in n_range:
        g = np.load("params/g_n"+str(n)+".npy")
        g_list.append(g)

    # Start figure
    plt.figure(figsize=(4,1.1))

    # Color scheme
    # Colormap experiment
    viridis = plt.cm.get_cmap('viridis')
    brkoff_col = "deeppink"

    # Generate the list of colors from the colormap
    color_indices = np.linspace(0,0.75,len(n_range)+1)
    colorpalette = [viridis(ci) for ci in color_indices]

    ## PART C / Transmission function
    for kN, (N,g) in enumerate(zip(n_range,g_list)):
        plt.plot(np.arange(1,N),g,marker=".",markersize=5,color="#af0002ff")
    # axs[1,0].legend(loc="upper left")

    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.2, hspace=0.5)  # or 0 for no spacing
    plt.savefig("figs/coupling.svg")
    plt.show()


def plot_transmission_only():
    """
    Plotting of transmission function for Fig 1
    """
    # Which n to plot
    n_range = np.array([20])
    g_list = []
    Heff_list = []

    for n in n_range:
        g = np.load("params/g_n"+str(n)+".npy")
        g_list.append(g)

    g_ref = g_list[-1][int(n_range[-1]/2)]

    for kN, n in enumerate(n_range):
        g = g_list[kN]
        g = g * (g_ref / g[int(n/2)])
        g_list[kN] = g
        H = np.diag(g,+1)+np.diag(g,-1)
        H=H.astype(complex)
        Heff_list.append(H)

    # Start figure
    plt.figure(figsize=(2,1.1))

    # Color scheme
    # Colormap experiment
    viridis = plt.cm.get_cmap('viridis')
    brkoff_col = "deeppink"

    # Generate the list of colors from the colormap
    color_indices = np.linspace(0,0.75,len(n_range)+1)
    colorpalette = [viridis(ci) for ci in color_indices]

    ## PART C / Transmission function
    E_range = np.linspace(-0.8,0.8,1000)

    for kN, (N,H) in enumerate(zip(n_range,Heff_list)):
        plt.plot(E_range/g_ref,np.vectorize(lambda x : T(x,H))(E_range),label=r"$N="+str(N)+"$",
                      color=colorpalette[kN],linestyle="-",linewidth=0.5)
    # axs[1,0].legend(loc="upper left")

    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.2, hspace=0.5)  # or 0 for no spacing
    plt.savefig("figs/trsm.svg")
    plt.show()


def to_black(color, t):
    r, g, b, a = color
    return (r * (1 - t), g * (1 - t), b * (1 - t), a)


def plot_cg_variance_parallel(times_list,N_list):
    """
    Visualization of the coarse-grained variance

    times_list  :   list of arrays "times" with dimension ~ [runs,times]
    N_list      :   list of N's corresponding to times
    """
    # Colormap experiment
    viridis = plt.cm.get_cmap('viridis')

    # Coarse-graining numbers
    n_cg_list       = np.unique((10**np.linspace(0,4,40)).astype(int))

    # Obtain central coupling
    g_list = np.zeros((len(N_list)))
    for kN, N in enumerate(N_list):
        g_list[kN] = np.load("params/g_n"+str(N)+".npy")[int((N-1)/2)]

    # Normalize times
    for k_times, times in enumerate(times_list):
        times *= (2*g_list[k_times] / np.pi)

    # Initialize plot
    plt.figure(figsize=(3.8,2.2))

    # Colors for fig (a)
    color_indices = np.linspace(0,0.75,3)
    colorpalette_a = [viridis(ci) for ci in color_indices[::-1]]

    # Initialize
    var_list        = np.zeros((len(n_cg_list)))
    avg_list        = np.zeros((len(n_cg_list)))
    var_list_err    = np.zeros((len(n_cg_list)))

    # Averages for normalization
    avgs = np.zeros((len(times_list)))
    
    # Generate the list of colors from the colormap
    color_indices = np.linspace(0,0.75,len(times_list))
    colorpalette = [viridis(ci) for ci in color_indices[::-1]]
    
    symbol_list = ["o","v","d","s"]

    for k_times, times in enumerate(times_list):
        # Determine CG variances
        for k, n_cg in enumerate(n_cg_list):
            deltaT_n_cg = np.array([])
            
            # Iterate through all runs
            for ell in range(times.shape[0]):
                # Coarse-grained times
                cg_times = times[ell,::n_cg]
                # Add differences to array
                deltaT_n_cg = np.concatenate((deltaT_n_cg,cg_times[1:]-cg_times[:-1]))
                
            var_list[k] = np.var(deltaT_n_cg)
            avg_list[k] = np.average(deltaT_n_cg)
            var_list_err[k] = np.sqrt(var_list[k]) * np.sqrt(4/(len(deltaT_n_cg)-2))

        # Determine avg. WT
        avgT = avg_list[0]
        avgs[k_times] = avgT

        # Determine current noise
        g = np.load("params/g_n"+str(N_list[k_times])+".npy")
        D = noise_vs_signal_residue(g,full=True)[0]

        # Plotting
        if k_times == 0:
            # Diffusive scaling
            plt.plot(n_cg_list,n_cg_list*var_list[0],color="black",label=r"iid")
            # Plot RMT prediction
            plt.semilogx(n_cg_list, 1/(np.pi)**2 * (np.log(0.5*np.pi*(n_cg_list)) + np.euler_gamma + 5/6),color="crimson",label="rmt")

        # Actual coarse-grained variance
        # plt.errorbar(n_cg_list,var_list * (avgs[0]**2/avgT**2),yerr=var_list_err,color=col_scheme_BR[k_times],linestyle="",marker=".",
        #             label=r"$\widehat{\rm Var}[T_n]$, $N="+str(N_list[k_times])+"$")        
        plt.errorbar(n_cg_list,var_list,yerr=var_list_err,color=colorpalette[k_times],linestyle="",markersize=4,marker=symbol_list[k_times],#marker=".",
                    label=r"$N="+str(N_list[k_times])+"$")
        
        # Plotting of asymptotic curve
        # plt.plot(n_cg_list[-25+5*k_times:], (np.pi / (2*g_list[k_times])) * D*avgT**3*n_cg_list[-25+5*k_times:],color=colorpalette[k_times],linestyle="solid",linewidth=1)       


    plt.ylim([0,2])

    plt.tight_layout()
    plt.savefig("figs/var_cg_SW.svg")#,dpi=600)
    plt.show()

    return None

def plot_varNT_comparisons(fresh=False):
    """
    Visualization of variance of Nt:
    - numerical simulation
    - from D calculation covariance
    - from Var Nt analytical bulk

        times:  shape ~ [runs,times]
        M   :   number of bootstrap samples
    """
    print("VarNt plotting function started.")

    # Setting
    N = 40

    # Load times and coupling
    times = np.load("data/tmp/times_parallel_n"+str(N)+".npy")
    _g     = np.load("params/g_n"+str(N)+".npy")
    g      = _g[int(N/2)]

    # Rescale times
    times *= (2*g)/np.pi

    def obtainVarNTPlottables(times_eval,t_range):
        """
        Subfunction to return VarNT values
        times_eval --> MC values
        t_range    --> Analytical predictions
        """
        # Cutter index
        idx_strt = np.argmax(t_range>=times_eval[0])

        times_num  = []
        varNT_num  = []
        k_num      = [0,1,N//2]

        print("Start numerical evaluation covariance matrix.")
        # Boundary current
        D, J = noise_vs_signal_COV_finite_time(_g,t_range*np.pi/(2*g))
        D_integral = cumulative_trapezoid(D,t_range*np.pi/(2*g),initial=0)

        times_num.append(t_range[idx_strt:])
        varNT_num.append(D_integral[idx_strt:])
        print("... boundary simulation complete!")


        # Bulk current
        for k in k_num[1:]:
            # Obtain diffusion constant
            D, J = bulk_current_COV_finite(_g,t_range*np.pi/(2*g),k=k)
            D_integral = cumulative_trapezoid(D,t_range*np.pi/(2*g),initial=0)

            times_num.append(t_range[idx_strt:])
            varNT_num.append(D_integral[idx_strt:])
            print(f"... offset k = {k:02d} simulation complete!")

        print("Start evaluation Monte Carlo sampling.")
        # Result array
        varN     = np.zeros((len(times_eval)))
        varN_err = np.zeros((len(times_eval)))

        # Pseudorrandomized sampling
        for kt, t in enumerate(times_eval):
            # All maximal times
            t_max = times[:,-1] - t
            t_min = times[:,0]
            num_bin = ((t_max-t_min)/t).astype(int)

            N_samp = []# np.array([])
            N_err_samp = []

            # Iterate over runs
            for k_run in range(times.shape[0]):
                # Numer of samples
                bins=np.arange(0,num_bin[k_run]+1)*t + t_min[k_run]
                hist, edges = np.histogram(times[k_run,:],bins=bins,range=(min(bins),max(bins)))

                N_samp.append(np.var(hist,ddof=1))
                N_err_samp.append(N_samp[-1] * 
                                (
                                    np.sqrt(4/(len(hist)-2)) 
                                    + 0.04 / ((2*np.pi/g))
                                ))
                    
            varN[kt] = np.average(np.array(N_samp))

            varN_err[kt] = np.average(np.array(N_err_samp))

        print("... statistical data binning complete.")

        return times_eval, varN, varN_err, times_num, varNT_num
    
    # Analytical prediction N=infty
    def f(a):
        # Bessel functions
        J0 = j0(a)
        J1 = j1(a)
        
        # Sine and Cosine integrals (SciPy returns both)
        Si, Ci = sici(2 * a)
        
        term1 = (a**2 / 4) * ((J0**2 + J1**2) - (J1 * J0) / a)
        term2 = (1 / np.pi**2) * (
            np.sin(a)**2 - a * Si + 0.5 * (np.log(2 * a) + np.euler_gamma - Ci)
        )
        
        return term1 + term2

    
    ##################
    # SEMILOG-X plot #
    ##################
    print("Plotting FIG-A")
    # Times array
    times_eval = 10**np.linspace(0,1.5,25)
    # Determine diffusion constants
    t_range = np.concatenate((np.linspace(0,1,100)[:-1],10**np.linspace(0,1.5,200)))

    if fresh:
        times_eval, varN, varN_err, times_num, varNT_num = obtainVarNTPlottables(times_eval,t_range)
        np.save("data/varNTplot/a/times_eval.npy",times_eval)
        np.save("data/varNTplot/a/varN.npy",varN)
        np.save("data/varNTplot/a/varN_err.npy",varN_err)
        np.save("data/varNTplot/a/times_num.npy",times_num)
        np.save("data/varNTplot/a/varNT_num.npy",varNT_num)

    else:
        times_eval = np.load("data/varNTplot/a/times_eval.npy")
        varN       = np.load("data/varNTplot/a/varN.npy")
        varN_err   = np.load("data/varNTplot/a/varN_err.npy")
        times_num  = np.load("data/varNTplot/a/times_num.npy")
        varNT_num  = np.load("data/varNTplot/a/varNT_num.npy")


    # Plotting
    plt.figure(figsize=(3.8,1.9))
    plt.errorbar(times_eval,varN,yerr=varN_err,linestyle="",marker="o",markersize=3.5,color="#440154ff",label="SSE")

    # 2 times Bulk
    # plt.plot(times_num[-1],2*varNT_num[-1],label=r"2 x offset="+str(k))
    plt.errorbar(times_eval,varN/2,yerr=varN_err/2,linestyle="",marker="v",markersize=3.5,color="#440154ff",label="SSE")

    # Iteration through different bulk-offsets
    color_palette = ["#440154ff","#5ec962ff","#21918cff"]
    linestyles    = ["--","solid","solid"]
    for kk, k in enumerate(np.array([0,1,N//2])):
        plt.plot(times_num[kk],varNT_num[kk],color=color_palette[kk],linestyle=linestyles[kk],label=r"offset="+str(k))

    plt.plot(times_num[0],f(np.pi*times_num[0]),color="#af0002ff")

    plt.xscale("log")
    # plt.yscale("log")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("figs/varNTa.svg")
    plt.show()

    ##################
    # LOGLOG plot    #
    ##################
    print("Plotting FIG-B")
    # Times array
    times_eval = 10**np.linspace(0,4,25)
    # Determine diffusion constants
    t_range = np.concatenate((np.linspace(0,1,100)[:-1],10**np.linspace(0,4,200)))

    if fresh:
        times_eval, varN, varN_err, times_num, varNT_num = obtainVarNTPlottables(times_eval,t_range)
        np.save("data/varNTplot/b/times_eval.npy",times_eval)
        np.save("data/varNTplot/b/varN.npy",varN)
        np.save("data/varNTplot/b/varN_err.npy",varN_err)
        np.save("data/varNTplot/b/times_num.npy",times_num)
        np.save("data/varNTplot/b/varNT_num.npy",varNT_num)

    else:
        times_eval = np.load("data/varNTplot/b/times_eval.npy")
        varN       = np.load("data/varNTplot/b/varN.npy")
        varN_err   = np.load("data/varNTplot/b/varN_err.npy")
        times_num  = np.load("data/varNTplot/b/times_num.npy")
        varNT_num  = np.load("data/varNTplot/b/varNT_num.npy")


    # Plotting
    plt.figure(figsize=(3.8,1.9))
    plt.errorbar(times_eval,varN,yerr=varN_err,linestyle="",marker="o",markersize=3.5,color="#440154ff",label="SSE")

    # 2 times Bulk
    # plt.plot(times_num[-1],2*varNT_num[-1],label=r"2 x offset="+str(k))
    # plt.errorbar(times_eval,varN/2,yerr=varN_err/2,linestyle="",marker="v",markersize=3.5,color="#440154ff",label="SSE")

    # Iteration through different bulk-offsets
    color_palette = ["#440154ff","#5ec962ff","#21918cff"]
    linestyles    = ["--","solid","solid"]
    for kk, k in enumerate(np.array([0,1,N//2])):
        plt.plot(times_num[kk],varNT_num[kk],color=color_palette[kk],linestyle=linestyles[kk],label=r"offset="+str(k))

    plt.plot(times_num[0],f(np.pi*times_num[0]),color="#af0002ff")

    plt.xscale("log")
    plt.yscale("log")
    # plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("figs/varNTb.svg")
    plt.show()

def plot_g2(N=80):
    """
    Visualization of the g2 function:

    N   :   system size (10, 20, 40, or 80)
    """
    print("G2 plotting function started.")

    # Load times and coupling
    _g     = np.load("params/g_n"+str(N)+".npy")
    g      = _g[int(N/2)]

    # Rescale times
    t_range = (np.linspace(0,5,600)[:-1])

    G2, J = g2(_g,t_range*np.pi/(2*g))

    
    #############
    # Main plot #
    #############
    plt.figure(figsize=(2.5,1.3))

    color_palette = ["#440154ff","#5ec962ff","#21918cff"]
    linestyles    = ["--","solid","solid"]

    plt.plot(t_range,G2,color=color_palette[0])

    plt.plot(t_range,1-np.sinc(t_range)**2,color=color_palette[1],linestyle=":")

    # plt.xscale("log")
    # plt.yscale("log")
    # plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("figs/g2.svg")
    plt.show()


def plotCovSkSk(times,N):
    """
    Visualization of Cov[S_i,S_i+k] where S_k is the time between
    tick k and k+1.

    times       :   times array (parallel)
    N           :   system size
    """
    # Obtain central coupling
    _g = np.load("params/g_n"+str(N)+".npy")[int((N-1)/2)]

    # Normalize times
    times *= (2*_g / np.pi)

    # Distance
    k_range = np.arange(1,26)

    covs = np.zeros((len(k_range)))
    errs = np.zeros((len(k_range)))

    # Iterate through all distances
    for kk, k in enumerate(k_range):
        tmp = []
        tmp_err = []

        # Iterate through all parallel runs
        for ell in range(times.shape[0]):
            # Determine tick time intervals
            Sk = times[ell,1:] - times[ell,:-1]

            SkA = Sk[k:]
            SkB = Sk[:-k]
            
            tmp.append(pearsonr(SkA,SkB).statistic)
        
        covs[kk] = np.average(tmp)
        errs[kk] = np.std(tmp)

    plt.figure(figsize=(3.8,1.9))

    color_palette = ["#440154ff","#5ec962ff","#21918cff"]
    linestyles    = ["--","solid","solid"]

    plt.errorbar(k_range,covs,yerr=errs,capsize=2,marker=".",color=color_palette[0])

    plt.tight_layout()
    plt.savefig("figs/SkSk.svg")
    plt.show()

    #########
    # INSET #
    #########
    plt.figure(figsize=(2.3,1.3))

    color_palette = ["#440154ff","#5ec962ff","#21918cff"]
    linestyles    = ["--","solid","solid"]

    plt.errorbar(k_range[::2],-covs[::2],yerr=errs[::2],capsize=2,marker=".",color=color_palette[0])
    plt.loglog(k_range,-covs[0]*k_range[0]**2/k_range**2,linestyle="--",color="black")
    plt.tight_layout()
    plt.savefig("figs/SkSk_inset.svg")
    plt.show()


def plot_cg_variance_parallel_LOGLOG(times_list,N_list):
    """
    Visualization of the coarse-grained variance

    times_list  :   list of arrays "times" with dimension ~ [runs,times]
    N_list      :   list of N's corresponding to times
    """
    # Colormap experiment
    viridis = plt.cm.get_cmap('viridis')

    # Coarse-graining numbers
    n_cg_list       = np.unique((10**np.linspace(0,4.4,40)).astype(int))

    # Obtain central coupling
    g_list = np.zeros((len(N_list)))
    for kN, N in enumerate(N_list):
        g_list[kN] = np.load("params/g_n"+str(N)+".npy")[int((N-1)/2)]

    # Normalize times
    for k_times, times in enumerate(times_list):
        times *= (2*g_list[k_times] / np.pi)

    # Initialize plot
    plt.figure(figsize=(3,2.5))

    # Colors for fig (a)
    color_indices = np.linspace(0,0.75,3)
    colorpalette_a = [viridis(ci) for ci in color_indices[::-1]]

    # Initialize
    var_list        = np.zeros((len(n_cg_list)))
    avg_list        = np.zeros((len(n_cg_list)))
    var_list_err    = np.zeros((len(n_cg_list)))

    # Averages for normalization
    avgs = np.zeros((len(times_list)))
    
    # Generate the list of colors from the colormap
    color_indices = np.linspace(0,0.75,len(times_list))
    colorpalette = [viridis(ci) for ci in color_indices[::-1]]

    symbol_list = ["o","v","d","s"]

    for k_times, times in enumerate(times_list):
        # Determine CG variances
        for k, n_cg in enumerate(n_cg_list):
            deltaT_n_cg = np.array([])
            
            # Iterate through all runs
            for ell in range(times.shape[0]):
                # Coarse-grained times
                cg_times = times[ell,::n_cg]
                # Add differences to array
                deltaT_n_cg = np.concatenate((deltaT_n_cg,cg_times[1:]-cg_times[:-1]))
                
            var_list[k] = np.var(deltaT_n_cg)
            avg_list[k] = np.average(deltaT_n_cg)
            var_list_err[k] = np.sqrt(var_list[k]) * np.sqrt(4/(len(deltaT_n_cg)-2))

        # Determine avg. WT
        avgT = avg_list[0]
        avgs[k_times] = avgT

        # Determine current noise
        g = np.load("params/g_n"+str(N_list[k_times])+".npy")
        D = noise_vs_signal_residue(g,full=True)[0]

        # Plotting
        if k_times == 0:
            # Diffusive scaling
            plt.plot(n_cg_list,n_cg_list*var_list[0],color="black",label=r"iid")
            # Plot RMT prediction
            plt.loglog(n_cg_list, 1/(np.pi)**2 * (np.log(0.5*np.pi*(n_cg_list)) + np.euler_gamma + 5/6),color="#af0002ff",label="rmt")

        # Actual coarse-grained variance
        # plt.errorbar(n_cg_list,var_list * (avgs[0]**2/avgT**2),yerr=var_list_err,color=col_scheme_BR[k_times],linestyle="",marker=".",
        #             label=r"$\widehat{\rm Var}[T_n]$, $N="+str(N_list[k_times])+"$")        
        plt.errorbar(n_cg_list,var_list,yerr=var_list_err,color=colorpalette[k_times],linestyle="",markersize=4,marker=symbol_list[k_times],#marker=".",
                    label=r"$N="+str(N_list[k_times])+"$")
        
        # Plotting of asymptotic curve
        plt.plot(n_cg_list[-30+5*k_times:], (np.pi / (2*g_list[k_times])) * D*avgT**3*n_cg_list[-30+5*k_times:],color=colorpalette[k_times],linestyle="dashed",linewidth=1)       

        def cross(Ay, Ax, By, Bx):
            y = np.union1d(Ay, By)
            d = np.interp(y, Ay, Ax) - np.interp(y, By, Bx)
            i = np.where(d[:-1] * d[1:] <= 0)[0][0]
            yc = y[i] - d[i] * (y[i+1] - y[i]) / (d[i+1] - d[i])
            return yc, np.interp(yc, Ay, Ax)
        
        print(cross((np.pi / (2*g_list[k_times])) * D*avgT**3*n_cg_list[-30+5*k_times:],n_cg_list[-30+5*k_times:],1/(np.pi)**2 * (np.log(0.5*np.pi*(n_cg_list)) + np.euler_gamma + 5/6),n_cg_list))

    # plt.ylim([0,2])

    plt.tight_layout()
    plt.savefig("figs/var_cg_loglog.svg")#,dpi=600)
    plt.show()

    return None


def plot_tilted_WTD(times):
    """
    Visualization
    (a) Plotting of the waiting time

    times_list  :   list of arrays "times" with dimension ~ [runs,times]
    N_list      :   list of N's corresponding to times
    """
    # Colormap experiment
    viridis = plt.cm.get_cmap('viridis')

    # Coarse-graining numbers
    n_cg_list       = np.unique((10**np.linspace(0,5,40)).astype(int))

    for step in [1,2,4]:
        # Initialize plot
        plt.figure(figsize=(2.5,1.3))

        # Plotting of part (a) / Waiting times
        deltaT_list = []
        for k_samp in range(times.shape[0]):
            deltaT_list.append(times[k_samp,step::step]-times[k_samp,:-step:step])

        # Colors for fig (a)
        color_indices = np.linspace(0,0.75,3)
        colorpalette_a = [viridis(ci) for ci in color_indices[::-1]]

        # Nearestneighbor waiting times
        deltaT_list = np.concatenate(deltaT_list)

        avgT = np.average(deltaT_list)
        deltaT_list /= avgT

        # Obtain Historgrams
        hist_y, hist_edge = np.histogram(deltaT_list,bins=50,range=(-0.02,3.2),density=True,)
        hist_x = hist_edge[:-1] + 0.5*(hist_edge[1]-hist_edge[0])

        # Wigner-Dyson fit
        def fun(x,p0,p1):
            return 32/np.pi**2 * (x/p0)**2 * np.exp(-4/np.pi * (x/p1)**2)
        popt, pcov = curve_fit(fun,hist_x,hist_y,np.array([1.0,1.0]),)
        x_range = np.linspace(0,1.2*max(hist_edge),1000)

        # Plot Fit
        if step==1:
            # Plot Wigner-Dyson prediction
            plt.plot(x_range,fun(x_range,*popt),
                    label=r"rmt",color="#af0002ff")
        plt.vlines(np.average(deltaT_list),ymin=-1,ymax=10,
                color="#66027eff",linewidth=1,
                linestyle="solid")#,label=r"$\tau$")
        plt.ylim(-0.005,1.2*np.max(hist_y))

        # Plot histogram unconditional
        plt.plot(hist_x,hist_y,
                linestyle="solid",
                marker="o",markersize=2.5,
                linewidth=1,color="#66027eff",#colorpalette_a[1],
                label=r"$p(t)$, $\tau$")

        # Plot conditional histograms
        times_post_slow = deltaT_list[1+np.argwhere(deltaT_list<np.average(deltaT_list))[:-1]]
        hist_y, hist_edge = np.histogram(times_post_slow,bins=30,density=True,)
        hist_x = hist_edge[:-1] + 0.5*(hist_edge[1]-hist_edge[0])
        plt.plot(hist_x,hist_y,
                linestyle="solid",
                marker="^",markersize=2.5,
                linewidth=1,color="#be006fff",#colorpalette_a[0],
                label=r"$p(t)_-$, $\tau_-$")
        plt.vlines(np.average(times_post_slow),ymin=-1,ymax=10,
                color="#be006fff",linewidth=1,
                linestyle="solid")#,label=r"$\tau_-$")

        times_post_fast = deltaT_list[1+np.argwhere(deltaT_list>np.average(deltaT_list))[:-1]]
        hist_y, hist_edge = np.histogram(times_post_fast,bins=30,density=True,)
        hist_x = hist_edge[:-1] + 0.5*(hist_edge[1]-hist_edge[0])
        plt.plot(hist_x,hist_y,
                marker="d",markersize=2.5,
                linestyle="solid",linewidth=1,color="#0d0b8aff",#colorpalette_a[2],
                label=r"$p(t)_+$, $\tau_+$")
        plt.vlines(np.average(times_post_fast),ymin=-1,ymax=10,
                color="#0d0b8aff",linewidth=1,
                linestyle="solid")#,label=r"$\tau_+$")

        plt.xlim(-0.1,3.1)

        plt.tight_layout()
        plt.savefig("figs/tiltedWTD"+str(step)+".svg")#,dpi=600)
        plt.show()

    return None

def plot_cg_variance_temps(times_list,s_list,N):
    """
    Plotting variance for different temperatures

    times_list  :   list of arrays "times" with dimension ~ [runs,times]
    s_list      :   list of entropies s
    N           :   fixed  N
    """
    # Length
    N = 20

    # List of times
    times_list = [
        np.load("data/tmp/times_parallel_n20.npy"),
        np.load("data/times_perturby/times_parallel_n20_s7.0.npy"),
        np.load("data/times_perturby/times_parallel_n20_s6.0.npy"),
        np.load("data/times_perturby/times_parallel_n20_s5.0.npy"),
    ]

    # Entropy list
    s_list = [
        np.inf,
        7.0,
        6.0,
        5.0,
    ]

    # Coarse-graining numbers
    n_cg_list       = np.unique((10**np.linspace(0,3,35)).astype(int))

    # Initialize plot
    fig, (ax1) = plt.subplots(1,1,figsize=(4,2))

    # Colormap experiment
    viridis = plt.cm.get_cmap('viridis')
    brkoff_col = "deeppink"

    # Generate the list of colors from the colormap
    color_indices = np.linspace(0,0.75,4)
    colorpalette = [viridis(ci) for ci in color_indices]

    # Key Color
    col_key = colorpalette[-2]

    print(col_key[:-1]*2)

    color_indices = np.linspace(0,0.75,len(times_list))
    colorpalette = [viridis(ci) for ci in color_indices]

    # Initialize
    var_list        = np.zeros((len(n_cg_list)))
    avg_list        = np.zeros((len(n_cg_list)))
    var_list_err    = np.zeros((len(n_cg_list)))

    # Averages for normalization
    avgs = np.zeros((len(times_list)))

    # Bulk coupling for RMT prediction
    g = np.load("params/g_n"+str(N)+".npy")

    for k_times, times in enumerate(times_list):
        # Determine CG variances
        for k, n_cg in enumerate(n_cg_list):
            deltaT_n_cg = np.array([])
            
            # Iterate through all runs
            for ell in range(times.shape[0]):
                # Coarse-grained times
                cg_times = times[ell,::n_cg]
                # Add differences to array
                deltaT_n_cg = np.concatenate((deltaT_n_cg,cg_times[1:]-cg_times[:-1]))
                
            var_list[k] = np.var(deltaT_n_cg)
            avg_list[k] = np.average(deltaT_n_cg)
            var_list_err[k] = var_list[k] * np.sqrt(4/(len(deltaT_n_cg)-2))

        # Determine avg. WT
        avgT = avg_list[0]
        avgs[k_times] = avgT

        # Rescale to match current for all
        var_list = var_list / avgT**2 #* (avgs[0]**2/avgT**2)
        var_list_err /= avgT**2

        # Plotting
        if k_times == 0:
            # Diffusive scaling
            # ax1.plot(n_cg_list,n_cg_list*var_list[0],color="black",linestyle="-.",label=r"iid")
            pass
        if k_times==len(times_list)-1:
            # Plot RMT prediction
            plt.semilogx(n_cg_list, 1/(np.pi)**2 * (np.log(0.5*np.pi*(n_cg_list)) + np.euler_gamma + 5/6),color="#af0002ff",label="rmt")

        # Plot ideal case
        ax1.errorbar(n_cg_list,var_list,yerr=var_list_err,
                    color=to_black(col_key,k_times / len(times_list)),
                    linestyle="",marker=".")
            
        # Plot diffusion constant
        _D, _J = noise_vs_signal_residue(g,full=True,gamma=1+np.exp(-s_list[k_times]))
        D_beta = 0.5 / np.cosh(s_list[k_times]/2)**2 + _D / _J * np.tanh(s_list[k_times]/2)**2

        idx_strt = [-14,-16,-19,-23,-30][k_times]
        ax1.plot(
            n_cg_list[idx_strt:],D_beta*n_cg_list[idx_strt:],
            color=to_black(col_key,k_times / len(times_list)),
            linestyle="dashed",linewidth=1
        )       
            
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$n$")
    ax1.set_ylabel(r"${\rm Var}[T_n]$")
    # ax1.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("figs/Fig_VarTemp.svg")#,dpi=300)
    plt.show()

    return None


def plot_cg_variance_EndMatter():
    """
    localization VS temp VS finite size
    """
    N = 40

    # Times list
    times_list = [
        # np.load("data/times_perturby/times_parallel_n20_s7.0.npy"),
        # np.load("data/times_perturby/times_parallel_n20_s7.0_debug.npy"),
        # np.load("data/times_perturby/times_parallel_n20_s8.0.npy"),
        # np.load("data/times_perturby/times_parallel_n20_s9.0.npy"),
        np.load("data/times_perturby/times_parallel_n40_s5.5.npy"),
        np.load("data/times_perturby/times_parallel_n40_16x100k_loc.npy"),
        # np.load("data/times_perturby/times_parallel_n40_s8.0.npy"),
        # np.load("data/times_perturby/times_parallel_n40_s9.0.npy"),
        np.load("data/tmp/times_parallel_n40.npy"),
    ]

    # Entropy!
    Sigma = 5.5

    # Coarse-graining numbers
    n_cg_list       = np.unique((10**np.linspace(0,4,40)).astype(int))

    # Initialize plot
    fig, (ax1) = plt.subplots(1,1,figsize=(3.8,2.2))

    # Colormap experiment
    viridis = plt.cm.get_cmap('viridis')
    brkoff_col = "deeppink"

    # Generate the list of colors from the colormap
    color_indices = np.linspace(0,0.75,4)
    colorpalette = [viridis(ci) for ci in color_indices]

    # Key Color
    col_key = colorpalette[-2]

    print(col_key[:-1]*2)

    color_indices = np.linspace(0,0.75,len(times_list)+1)
    colorpalette = [viridis(ci) for ci in color_indices]
    colorpalette = ["#2c9e56ff","#208385ff","#3b528bff"] # "#a51b20ff","#ba3091ff","#8845adff"

    symbol_list = ["o","v","d","s"]

    # Initialize
    var_list        = np.zeros((len(n_cg_list)))
    avg_list        = np.zeros((len(n_cg_list)))
    var_list_err    = np.zeros((len(n_cg_list)))

    # Averages for normalization
    avgs = np.zeros((len(times_list)))

    # Bulk coupling for RMT prediction
    g = np.load("params/g_n"+str(N)+".npy")
    g_loc = np.load("params/perturby/g_n"+str(N)+".npy")
    g_bulk = g[int((N-1)/2)]

    ########################
    # Diffusion constant D #
    ########################

    # Determine diffusion constants
    D_ideal = noise_vs_signal_residue(g)
    D_loc   = noise_vs_signal_residue(g_loc)

    _D, _J = noise_vs_signal_residue(g,full=True,gamma=1+np.exp(-Sigma))
    D_beta = 0.5 / np.cosh(Sigma/2)**2 + _D / _J * np.tanh(Sigma/2)**2

    D_list = [D_beta,D_loc,D_ideal]

    # Setup for scaling calculation
    varThreshold = 30.0
    n_breakoff = np.zeros((len(times_list)))

    for k_times, times in enumerate(times_list):
        # Determine CG variances
        for k, n_cg in enumerate(n_cg_list):
            deltaT_n_cg = np.array([])
            
            # Iterate through all runs
            for ell in range(times.shape[0]):
                # Coarse-grained times
                cg_times = times[ell,::n_cg]
                # Add differences to array
                deltaT_n_cg = np.concatenate((deltaT_n_cg,cg_times[1:]-cg_times[:-1]))
                
            var_list[k] = np.var(deltaT_n_cg)
            avg_list[k] = np.average(deltaT_n_cg)
            var_list_err[k] = var_list[k] * np.sqrt(4/(len(deltaT_n_cg)-2))

        # Determine avg. WT
        avgT = avg_list[0]
        avgs[k_times] = avgT

        print(avgT / (np.pi/(2*np.load("params/g_n20.npy")[10])))

        # Rescale to match current for all
        var_list = var_list / avgT**2 #* (avgs[0]**2/avgT**2)
        var_list_err /= avgT**2

        # Plotting
        if k_times == 0:
            # Diffusive scaling
            # ax1.plot(n_cg_list,n_cg_list*var_list[0],color="black",linestyle="-.",label=r"iid")
            pass
        if k_times==len(times_list)-1:
            # Plot RMT prediction
            plt.loglog(n_cg_list, 1/(np.pi)**2 * (np.log(0.5*np.pi*(n_cg_list)) + np.euler_gamma + 5/6),color="#af0002ff",label="rmt")

        # Actual coarse-grained variance
        if True: #(k_times<len(times_list)-1) and k_times % 2 ==0:
            ax1.errorbar(n_cg_list,var_list,yerr=var_list_err,
                         color=colorpalette[k_times],
                         linestyle="",
                         markersize=4,marker=symbol_list[k_times],
                        #  marker=".",
                         label=str(k_times))

            idx_strt = -25+5*k_times
            print(idx_strt)
            plt.plot(n_cg_list[idx_strt:],
                     D_list[k_times] *n_cg_list[idx_strt:],color=colorpalette[k_times],linestyle="dashed",linewidth=1)       

    # ax1.hlines(varThreshold,xmin=min(n_cg_list),xmax=max(n_cg_list),color=brkoff_col,linewidth=1,linestyle=":")
            
    # ax1.set_ylim([0.1,1.4])
    ax1.set_xlabel(r"$n$")
    ax1.set_ylabel(r"${\rm Var}[T_n]$")
    # ax1.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("figs/cross_overs.svg")#,dpi=300)
    plt.show()

    return None

def plot_dispersion(ss_list):
    """
    Saves dispersion relation figure
    
    Input:
        states  :   shape [num_states,N,N]
        H       :   Hamiltonian
    """

    N_to_col = {
        10 : "#5ec962ff",
        20 : "#21918cff",
        40 : "#3b528bff",
        100 : "#440154ff",
    }

    k_list   = []
    occ_list = []

    for ss in ss_list:
        # System size
        N = ss.shape[-1]
        N_cut = 3

        # Get Hamiltonian
        g = np.load("params/g_n"+str(N)+".npy")
        g_bulk = g[int(N/2)]
        g = g[N_cut:-N_cut]

        H = (np.diag(g,+1) + np.diag(g,-1)).astype(np.complex128)

        # Momentum range
        k = 2*np.pi * (np.arange(N-2*N_cut))/(N-2*N_cut) - np.pi

        # Obtain matrices
        v = 1 / np.sqrt(N-2*N_cut) * np.exp(-1j*np.outer(np.arange(N-2*N_cut),k))
        vinv = v.T.conj()

        # Get occupation probabilities
        occ = np.real(np.diag(v@ss[N_cut:-N_cut,N_cut:-N_cut]@vinv))
        ev = np.real(np.diag(v@H@vinv))

        k_list.append(k)
        occ_list.append(occ)

    viridis = plt.cm.get_cmap('viridis')
    color_indices = np.linspace(0,0.99,1+len(ss_list))
    colorpalette = [viridis(ci) for ci in color_indices]

    fig, ax1 = plt.subplots(figsize=(4,1.7))
    ax2 = ax1.twinx() 

    # ax1.plot(k,ev,color=colorpalette[1])#,label=r"$E(k)$")

    k_cos = np.linspace(-np.pi,np.pi,300)
    ax1.plot(k_cos,-np.cos(k_cos),color="black")

    # Ideal case
    k_id = np.array([-np.pi,0,0,np.pi])
    p_id = np.array([0,0,1,1.])
    ax2.plot(k_id,p_id,color="#af0002ff")
    ax2.plot([-np.pi,-np.pi],[0,0.5],linestyle="--",color="#af0002ff")
    ax2.plot([np.pi,np.pi],[1,0.5],linestyle="--",color="#af0002ff")


    # Iteration through ss
    for k_enum, (k, occ) in enumerate(zip(k_list,occ_list)):
        colN = N_to_col[np.shape(ss_list[k_enum])[0]]
        ax2.plot(k,occ,marker="",color=colN)#,label=r"$E(k)$")

    ax1.set_xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],[r"$-\pi$",r"$-\pi/2$",r"$0$",r"$\pi/2$",r"$\pi$"])

    plt.tight_layout()
    plt.savefig("figs/FigSM1_Dispersion.svg")#,dpi=600)
    plt.show()