import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov
from scipy.integrate import solve_ivp

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

    # Current correlation
    corrs = np.zeros(shape=(len(t_range)))
    for i in range(len(t_range)):
        sigmat_i = np.reshape(solSigma.y[:, i], (N, N))
        corrs[i] = np.real_if_close(J * (sigmat_i[-1, -1] - fB))
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


if __name__ == "__main__":
    N = 100
    t_range = np.concatenate(
        (np.linspace(0, 1, 100)[:-1], 10 ** np.linspace(0, 1.5, 200))
    )
    end_of_chain(N, t_range)
    in_chain(N, 0, t_range)
    in_chain(N, 1, t_range)
    in_chain(N, 10, t_range)
    in_chain(N, N // 2, t_range)
    in_chain(N, N - 10, t_range)
    in_chain(N, N - 1, t_range)
