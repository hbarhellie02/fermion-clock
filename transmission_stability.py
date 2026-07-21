import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import eig
from scipy.integrate import quad
from transmission_clock import *

def anderson_localization_error(N_list,eps_range,Nsamp=1000,fresh=False):
    """
    Input:
        N   :   number of sites
        eps_range: n
    """
    N_list = np.atleast_1d(N_list)

    if not fresh:
        fano = np.load("data/stability/anderson_loc.npy")

        # Check correctness of dimensions
        if not ((fano.shape[0] == len(N_list)) and fano.shape[1] == len(eps_range)):
            print("Error in matching dimensions")
            return None
        else:
            pass

    else:
        # Prepare results [eps,samp]
        fano = np.zeros((len(N_list),len(eps_range),Nsamp))

        for kN, N in enumerate(N_list):
            # Load couplings
            g = np.load("params/g_n"+str(N)+".npy")

            D, J = noise_vs_signal_residue(g,full=True)

            W_range = g[int(N/2)] * eps_range

            # Cyclce through errors
            for k, W in enumerate(W_range):
                # Prepare random errors
                # -eps < err < eps
                err_factor = W*(np.random.random(size=(Nsamp,len(g)+1)) - 0.5)

                for m in range(Nsamp):
                    _D, _J = noise_vs_signal_residue(g,w=err_factor[m,:],full=True)
                    fano[kN,k,m] = _D/J

                print("eps "+str(np.round(W,3))+" complete.")

        np.save("data/stability/anderson_loc.npy",fano)

    # Statistics
    fano_avg = np.average(fano,axis=2)
    fano_dev = np.std(fano,axis=2,ddof=1)

    # Visualization
    # Color scheme
    # Colormap experiment
    viridis = plt.cm.get_cmap('viridis')
    brkoff_col = "deeppink"

    # Generate the list of colors from the colormap
    color_indices = np.linspace(0,0.75,len(N_list))
    colorpalette = [viridis(ci) for ci in color_indices[::-1]]


    # Plot D/J
    plt.figure(figsize=(4,2))
    for kN, N in enumerate(N_list):
        plt.errorbar(eps_range,fano_avg[kN,:],yerr=fano_dev[kN,:],capsize=3,marker=".",
                     color=colorpalette[kN])
    
        # Print critical eps value
        if N==20 or N==40:
            tol = 2.0

            g = np.load("params/g_n"+str(N)+".npy")
            fano_true = noise_vs_signal_residue(g)

            k_crit = np.max(np.argwhere(fano_avg[kN,:]<=tol*fano_true))
            alph = (tol*fano_true-fano_avg[kN,k_crit])/(fano_avg[kN,k_crit+1]-fano_avg[kN,k_crit])
            eps_crit = eps_range[k_crit] +  alph*(eps_range[k_crit+1]-eps_range[k_crit])

            print("eps_crit : "+str(eps_crit)+", N = "+str(N))

            plt.hlines(2*fano_true,min(eps_range),max(eps_range),color=colorpalette[kN],alpha=0.3,linestyle="--")
            plt.vlines(eps_crit,fano_true/2,2*fano_true,color="red",linestyle=":")
            plt.plot(eps_crit,2*fano_true,marker="*",color="red",markersize=6)

    
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("figs/loc_anderson.svg")
    plt.show()

    # Plot D/J log log for fitting
    plt.figure(figsize=(4,2))

    offsets = np.zeros((len(N_list)))
    alphas  = np.zeros((len(N_list)))

    for kN, N in enumerate(N_list):
        # Determine 'true' fano
        g = np.load("params/g_n"+str(N)+".npy")
        fano_true = noise_vs_signal_residue(g)

        plt.errorbar(eps_range,fano_avg[kN,:]-fano_true,
                     yerr=fano_dev[kN,:],
                     capsize=3,
                     marker=".",
                     linestyle="",
                     color=colorpalette[kN])
        
        # Fit cutoff
        k = int(len(eps_range)/3)
        print(k)

        z, cov = np.polyfit(np.log(eps_range[:k]),np.log((fano_avg[kN,:k]-fano_true)),deg=1,cov=True)
        
        plt.loglog(eps_range,np.exp(z[1])*eps_range**z[0],
                   color=colorpalette[kN])
        
        offsets[kN] = np.exp(z[1])
        alphas[kN] = (z[0])
        print("N="+str(N)+", alph="+str(np.round(z[0],3)), "err_alph="+str(np.round(np.sqrt(cov[0,0]),3)))
    
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("figs/loc_loglog_anderson.svg")
    plt.show()

    print("avg alpha="+str(np.round(np.average(alphas),3))+", std dev="+str(np.round(np.std(alphas),3)))

    # Plot offset scaling
    plt.figure(figsize=(1.8,1.4))
    plt.plot(N_list,offsets,marker=".",linestyle="",color="#af0002ff")

    z, cov = np.polyfit(np.log(N_list),np.log(offsets),deg=1,cov=True)

    plt.plot(N_list,np.exp(z[1])*N_list**z[0],marker="",linestyle="--",color="#af0002ff")
    print("beta="+str(np.round(z[0],3)), "err_beta="+str(np.round(np.sqrt(cov[0,0]),3)))


    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("figs/offset_anderson.svg")
    plt.show()


def localization_error(N_list,eps_range,Nsamp=1000,fresh=False):
    """
    Input:
        N   :   number of sites
        eps_range: n
    """
    N_list = np.atleast_1d(N_list)

    if not fresh:
        fano = np.load("data/stability/loc.npy")

        # Check correctness of dimensions
        if not ((fano.shape[0] == len(N_list)) and fano.shape[1] == len(eps_range)):
            print("Error in matching dimensions")
            return None
        else:
            pass

    else:
        # Prepare results [eps,samp]
        fano = np.zeros((len(N_list),len(eps_range),Nsamp))

        for kN, N in enumerate(N_list):
            # Load couplings
            g = np.load("params/g_n"+str(N)+".npy")

            g_bulk = g[int(N/2)]

            D, J = noise_vs_signal_residue(g,full=True)

            # Cyclce through errors
            for k, eps in enumerate(eps_range):
                # Prepare random errors
                err_factor = g_bulk * eps * ( np.random.random(size=(Nsamp,len(g)))-0.5 )

                for m in range(Nsamp):
                    _g = g + err_factor[m,:]

                    _D, _J =  noise_vs_signal_residue(_g,full=True)

                    fano[kN,k,m] = _D/_J

                print("eps "+str(np.round(eps,3))+" complete.")

        np.save("data/stability/loc.npy",fano)


    # Statistics
    fano_avg = np.average(fano,axis=2)
    fano_dev = np.std(fano,axis=2,ddof=1)

    # Visualization
    # Color scheme
    # Colormap experiment
    viridis = plt.cm.get_cmap('viridis')
    brkoff_col = "deeppink"

    # Generate the list of colors from the colormap
    color_indices = np.linspace(0,0.75,len(N_list))
    colorpalette = [viridis(ci) for ci in color_indices[::-1]]

    # Plot D/J
    plt.figure(figsize=(4,2))
    for kN, N in enumerate(N_list):
        plt.errorbar(eps_range,fano_avg[kN,:],yerr=fano_dev[kN,:],capsize=3,marker=".",
                     color=colorpalette[kN])
        
        # Print critical eps value
        if N==20 or N==40:
            tol = 2.0

            g = np.load("params/g_n"+str(N)+".npy")
            fano_true = noise_vs_signal_residue(g)

            k_crit = np.max(np.argwhere(fano_avg[kN,:]<=tol*fano_true))
            alph = (tol*fano_true-fano_avg[kN,k_crit])/(fano_avg[kN,k_crit+1]-fano_avg[kN,k_crit])
            eps_crit = eps_range[k_crit] +  alph*(eps_range[k_crit+1]-eps_range[k_crit])

            print("eps_crit : "+str(eps_crit)+", N = "+str(N))

            plt.hlines(2*fano_true,min(eps_range),max(eps_range),color=colorpalette[kN],alpha=0.3,linestyle="--")
            plt.vlines(eps_crit,fano_true/2,2*fano_true,color="red",linestyle=":")
            plt.plot(eps_crit,2*fano_true,marker="*",color="red",markersize=6)

    
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("figs/loc.svg")
    plt.show()

    # Plot D/J log log for fitting
    plt.figure(figsize=(4,2))

    offsets = np.zeros((len(N_list)))
    alphas  = np.zeros((len(N_list)))

    for kN, N in enumerate(N_list):
        # Determine 'true' fano
        g = np.load("params/g_n"+str(N)+".npy")
        fano_true = noise_vs_signal_residue(g)

        plt.errorbar(eps_range,fano_avg[kN,:]-fano_true,
                     yerr=fano_dev[kN,:],
                     capsize=3,
                     marker=".",
                     linestyle="",
                     color=colorpalette[kN])
        
        # Fit cutoff
        k = int(len(eps_range)/3)
        print(k)

        z, cov = np.polyfit(np.log(eps_range[:k]),np.log((fano_avg[kN,:k]-fano_true)),deg=1,cov=True)
        
        plt.loglog(eps_range,np.exp(z[1])*eps_range**z[0],
                   color=colorpalette[kN])
        
        offsets[kN] = np.exp(z[1])
        alphas[kN] = (z[0])
        print("N="+str(N)+", alph="+str(np.round(z[0],3)), "err_alph="+str(np.round(np.sqrt(cov[0,0]),3)))
    
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("figs/loc_loglog.svg")
    plt.show()

    print("avg alpha="+str(np.round(np.average(alphas),3))+", std dev="+str(np.round(np.std(alphas),3)))

    # Plot offset scaling
    plt.figure(figsize=(1.8,1.4))
    plt.plot(N_list,offsets,marker=".",linestyle="",color="#af0002ff")

    z, cov = np.polyfit(np.log(N_list),np.log(offsets),deg=1,cov=True)

    plt.plot(N_list,np.exp(z[1])*N_list**z[0],marker="",linestyle="--",color="#af0002ff")
    print("beta="+str(np.round(z[0],3)), "err_beta="+str(np.round(np.sqrt(cov[0,0]),3)))


    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("figs/offset.svg")
    plt.show()


eps_range = 10**np.linspace(-5,-1,25)
N_list = np.array([10,20,40,80])

# COUPLING DISORDER
eps_range = 10**np.linspace(-5,-1,25)
localization_error([10,20,40,80],eps_range)#,fresh=True)

# ANDERSON LOCALIZATION
eps_range = 10**np.linspace(-5,-1,25)
anderson_localization_error([10,20,40,80],eps_range)#,fresh=True)
