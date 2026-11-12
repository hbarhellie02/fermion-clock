from types import TracebackType

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov
from scipy.integrate import solve_ivp, cumulative_simpson

from scipy.special import j0, j1, sici
from transmission_clock import T, getGaussianSetup, getSS_covariance

plt.rcParams.update(
    {
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 12,
    }
)


def end_of_chain(N, t_range, plot=True):
    """
    return time dependent current-current correlation at the end of the chain
    """
    g = np.load("params/g_n" + str(N) + ".npy")
    N, fA, fB, Pi1, PiN, K, F = getGaussianSetup(g, s=np.inf)

    # Get SS
    C = getSS_covariance(g, s=np.inf)

    # Get current
    J = C[-1, -1] - fB

    # Get CSigma
    CSigma = (
        1
        / J
        * (
            fA * (C * C[-1, -1] - C @ PiN @ C)
            - fB * ((np.eye(N) - C) @ PiN @ (np.eye(N) - C) + C * (1 - C[-1, -1]))
        )
    )

    # ODE
    def func(t, y):
        return np.ravel(
            K @ np.reshape(y, (N, N)) + np.reshape(y, (N, N)) @ (K.conj().T) + F
        )

    solSigma = solve_ivp(
        func, (min(t_range), max(t_range)), np.ravel(CSigma), t_eval=t_range
    )
    # dynamical activity
    A = (1 - 2 * fB) * C[-1, -1] + fB
    # Current correlation
    corrs = np.zeros(shape=(len(t_range)))
    for i in range(len(t_range)):
        sigmat_i = np.reshape(solSigma.y[:, i], (N, N))
        corrs[i] = np.real_if_close(J * (sigmat_i[-1, -1] - fB) - J**2)
        if i == 0:
            corrs[i] += np.real_if_close(A)
    if plot:
        fig, ax = plt.subplots()
        ax.set_title("Current correlation at the end of chain")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\mathbb{E}[I(t)I(0)]$")
        ax.plot(t_range, corrs)
        plt.savefig("curr-end-correlations.svg")
    return corrs


def in_chain(N, k, t_range, plot=True):
    """
    return time dependent current-current correlation at site `N-k` of the chain
    """
    g = np.load("params/g_n" + str(N) + ".npy")
    N, fA, fB, Pi1, PiN, K, F = getGaussianSetup(g, s=np.inf)

    # Bulk coupling
    gBulk = g[-k]

    # Set current operator
    J = np.zeros((N, N), dtype=np.complex128)
    # J[N//2+1,N//2] = -1j * gBulk
    # J[N//2,N//2+1] = +1j * gBulk
    J[-k, -k - 1] = -1j * gBulk
    J[-k - 1, -k] = +1j * gBulk

    # Get SS
    C = getSS_covariance(g, s=np.inf)

    # Get current
    _J = np.trace(J @ C)

    # Get CSigma
    CSigma = 1 / _J * (C @ J - C @ J @ C + C * _J)

    # ODE
    def func(t, y):
        return np.ravel(
            K @ np.reshape(y, (N, N)) + np.reshape(y, (N, N)) @ (K.conj().T) + F
        )

    solSigma = solve_ivp(
        func, (min(t_range), max(t_range)), np.ravel(CSigma), t_eval=t_range
    )

    # Current correlation
    corrs = np.zeros(shape=(len(t_range)))
    for i in range(len(t_range)):
        sigmat_i = np.reshape(solSigma.y[:, i], (N, N))
        corrs[i] = np.real_if_close(_J * np.trace(J @ sigmat_i) - _J**2)
    if plot:
        fig, ax = plt.subplots()
        ax.set_title(f"Current correlation at site ${N - k}/{N}$")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\langle\langle j_k(t) j_k(0)\rangle\rangle$")
        ax.plot(t_range, corrs)
        plt.savefig(f"curr-{N - k}-correlations.svg")
    return corrs


def integrate_correlator(corrs, t_range):
    """
    Calculate the time-dependent variance given a correlator time series
    """
    first_integral = cumulative_simpson(corrs, x=t_range, initial=0)
    vars = cumulative_simpson(first_integral, x=t_range, initial=0)

    return vars


def f(a):
    # Bessel functions
    J0 = j0(a)
    J1 = j1(a)

    # Sine and Cosine integrals (SciPy returns both)
    Si, Ci = sici(2 * a)

    term1 = (a**2 / 4) * ((J0**2 + J1**2) - (J1 * J0) / a)
    term2 = (1 / np.pi**2) * (
        np.sin(a) ** 2 - a * Si + 0.5 * (np.log(2 * a) + np.euler_gamma - Ci)
    )

    return term1 + term2


if __name__ == "__main__":
    N = 100

    g_list = np.load("params/g_n" + str(N) + ".npy")
    g = g_list[N // 2]
    t_range = (
        np.concatenate((np.linspace(0, 1, 100)[:-1], 10 ** np.linspace(0, 1.5, 200)))
        * np.pi
        / (2 * g)
    )
    corr_N_class = end_of_chain(N, t_range)
    corr_N_quant = in_chain(N, 0, t_range)
    corr_N_1 = in_chain(N, 1, t_range)
    in_chain(N, 10, t_range)
    corr_N_half = in_chain(N, N // 2, t_range)
    in_chain(N, N - 10, t_range)
    in_chain(N, N - 1, t_range)
    vars_N_c = integrate_correlator(corr_N_class, t_range)
    vars_N_q = integrate_correlator(corr_N_quant, t_range)
    vars_N_1 = integrate_correlator(corr_N_1, t_range)
    vars_N_half = integrate_correlator(corr_N_half, t_range)
    fig, ax = plt.subplots()
    ax.loglog(t_range, -vars_N_c, label="N, Jump op.")
    ax.loglog(t_range, vars_N_q, label="N, curr. corr.")
    ax.loglog(t_range, vars_N_half, label="N/2, curr. corr.")
    ax.loglog(t_range, f(t_range), label="analyt.")
    ax.legend()
    ax.set_title("Variances from integrated correlators, for various sites")
    ax.set_ylabel(r"Var[$N_t$]")
    ax.set_xlabel(r"$J t$")
    plt.savefig("vars_N_classical_vs_quantum.svg")
