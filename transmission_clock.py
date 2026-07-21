import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import eig, solve_continuous_lyapunov
from scipy.integrate import quad, solve_ivp

def H(g):
    """
    g   :   NN-couplings, len = n-1

    returns Hamiltonian with size n x n
    """
    return (np.diag(g,-1) + np.diag(g,+1)).astype(complex)

def T(x,H):
    """
    x   :   evaluation parameter --> vectorized
    H   :   Hamiltonian

    returns Transmission function |<0|(H_eff-x)^(-1)|1>|^2
    """
    # Vectorize relative to x
    x = np.atleast_1d(x)

    # Params
    n = H.shape[0]

    # Sinks
    projL = np.zeros_like(H)
    projR = np.zeros_like(H)
    projL[0,0] = 1.
    projR[-1,-1] = 1.
    
    # Effective Hamiltonian
    H_eff = H - 0.5j*projL - 0.5j*projR

    # Solve LE for inverse
    ketL = np.zeros((n),dtype=complex)
    ketL[0] = 1.0

    # Determine result
    res = np.zeros_like(x)
    for i, xi in enumerate(x):
        Y = np.linalg.solve(H_eff - xi*np.eye(n),ketL)
        res[i] = np.abs(Y[-1])**2
    
    # Return
    return res

def noise_vs_signal_residue(g,full=False,gamma=1.0,w=np.array([0.])):
    """
    g   :   couplings

    return noise vs signal, using residue calculus method
    """
    # Set effective Hamiltonian
    H = np.diag(g,+1) + np.diag(g,-1)
    H = H.astype(complex)
    H[0,0] = -0.5j*gamma
    H[-1,-1] = -0.5j*gamma

    # Add diagonal if necessary
    if len(w) == (len(g)+1):
        H += np.diag(w)

    # Projectors
    projL = np.zeros_like(H)
    projR = np.zeros_like(H)
    projL[0,0] = 1.
    projR[-1,-1] = 1.

    # Diagonalization
    # Recall: v[i] ~ vl[:,i] ~ vr[:,i]
    w, vl, vr = eig(H,left=True,right=True)
    vl = vl.conj()

    # Obtain coeff
    c1 = (vl.T).dot(projL.dot(vl.conj()))
    c2 = ((vr.conj().T).dot(projR.dot(vr))).T
    innards = np.sum(vl*vr,axis=0)
    cnorm = np.outer(innards,innards.conj())

    # Calculate matrix element
    C = c1*c2 / cnorm

    # Determine evals
    n = H.shape[0]
    w_matrix = np.outer(w,np.ones((n)))
    c3 = w_matrix.conj().T - w_matrix

    # Determine current
    intT = np.real(1j * np.sum(C/c3))*gamma**2

    # Determine noise
    # Weird factor
    weird = w_matrix.T - w_matrix + np.eye(n)
    weird = 1 / weird - np.eye(n)

    a_1 = np.sum(C/c3,axis=1)
    a_0 = np.sum(C/c3**2,axis=1) - np.sum(weird.dot(C) / c3,axis=1) # np.sum(C*weird,axis=0) * np.sum(weird,axis=1)

    intT2 = -np.real(2 * 1j * np.inner(a_1,a_0)) * gamma**4

    if full:
        # Return S, I
        return (intT - intT2), intT
    
    else:
        # Return S/I
        return (intT - intT2)/intT # -intT2, intT # (intT - intT2)/intT
    
def getGaussianSetup(g,s):
    """
    g   :   couplings
    s   :   entropy factor

    Return N, fA, fB, Pi1, PiN, K, F
    """
    # Size
    N = len(g)+1

    # Check reversible
    if s<np.inf:
        # Set fermi functions
        fA = 1/(1+np.exp(-s))
        fB = np.exp(-s)/(1+np.exp(-s))
    else:
        fA = 1.0
        fB = 0.0

    # Define relevant operators
    Pi1 = np.zeros((N),dtype=np.complex128)
    Pi1[0] = 1.0
    Pi1 = np.diag(Pi1)
    
    PiN = np.zeros((N),dtype=np.complex128)
    PiN[-1] = 1.0
    PiN = np.diag(PiN)

    # Lyapunov ops
    K = 1j*(np.diag(g,+1)+np.diag(g,-1)).astype(np.complex128)
    K[0,0] = -0.5
    K[-1,-1] = -0.5

    F = fA*Pi1 + fB*PiN

    # Return
    return N, fA, fB, Pi1, PiN, K, F


def getSS_covariance(g,s=np.inf):
    """
    Return covariance matrix for stationary state

    Input:  g (couplings), s (entropy)
    """
    # Setup
    N, fA, fB, Pi1, PiN, K, F = getGaussianSetup(g,s)

    # Equation: KC + CK^dag + F = 0
    C = solve_continuous_lyapunov(K,-F)

    return C

def noise_vs_signal_COV(g,s=np.inf):
    """
    Return master equation expression for current and noise
    D = K + int dt (tr[J exp(Lt) J rho] - J^2)
    J = tr[J rho]

    Input:  g (couplings), s (entropy)
    """
    # Setup (note, fL = fA = 1-fR, fR = fB = 1-fA)
    N, fA, fB, Pi1, PiN, K, F = getGaussianSetup(g,s)

    # Get SS
    C = getSS_covariance(g,s)

    # Get current
    J = C[-1,-1] - fB

    # Get CSigma
    CSigma = 1/J * (fA * (C * C[-1,-1] - C@PiN@C) 
                    - fB * ((np.eye(N)-C)@PiN@(np.eye(N)-C) + C*(1-C[-1,-1])))
    
    # Get X: KX + XK^dag = CSigma   -> X = K^-1 Csigma
    X = solve_continuous_lyapunov(K,CSigma)

    # Get Y: KY + YK^dag + Css = 0  -> Y = -K^-1 Css
    Y = solve_continuous_lyapunov(K,-C)

    # Get noise
    D = (fA-fB)*C[-1,-1] + fB - 2*J*(X[-1,-1] + Y[-1,-1])

    # Return
    return D.real, J.real

def noise_vs_signal_COV_FW_ONLY(g,s=np.inf):
    """
    Return master equation expression for current and noise
    where J -> ONLY R OUT CURRENT
    D = K + int dt (tr[J exp(Lt) J rho] - J^2)
    J = tr[J rho]

    Input:  g (couplings), s (entropy)
    """
    # Setup (note, fL = fA = 1-fR, fR = fB = 1-fA)
    N, fA, fB, Pi1, PiN, K, F = getGaussianSetup(g,s)

    # Get SS
    C = getSS_covariance(g,s)

    # Get current
    J = fA * C[-1,-1]

    # Get CSigma
    CSigma = 1/J * (fA * (C * C[-1,-1] - C@PiN@C))
    
    # Get X: KX + XK^dag = CSigma
    X = solve_continuous_lyapunov(K,CSigma)

    # Get Y: KY + YK^dag + Css = 0
    Y = solve_continuous_lyapunov(K,-C)

    # Get noise
    D = J - 2 * J * fA * (X[-1,-1] + Y[-1,-1])

    # Return
    return D, J

def noise_vs_signal_COV_finite_time(g,t_range):
    """
    Return master equation expression for current and noise
    D = K + int dt (tr[J exp(Lt) J rho] - J^2)
    J = tr[J rho]

    Input:  g (couplings), t_range, s (entropy)
    """
    # Setup (note, fL = fA = 1-fR, fR = fB = 1-fA)
    N, fA, fB, Pi1, PiN, K, F = getGaussianSetup(g,s=np.inf)

    # Get SS
    C = getSS_covariance(g,s=np.inf)

    # Get current
    J = C[-1,-1] - fB

    # Get CSigma
    CSigma = 1/J * (fA * (C * C[-1,-1] - C@PiN@C) 
                    - fB * ((np.eye(N)-C)@PiN@(np.eye(N)-C) + C*(1-C[-1,-1])))

    # ODE
    def func(t,y):
        return np.ravel(K@np.reshape(y,(N,N)) + np.reshape(y,(N,N))@(K.conj().T))

    solSigma = solve_ivp(func,(min(t_range),max(t_range)),np.ravel(CSigma),t_eval=t_range)
    solSS    = solve_ivp(func,(min(t_range),max(t_range)),np.ravel(C),t_eval=t_range)

        
    # Get X: KX + XK^dag = CSigma - Css
    X = solve_continuous_lyapunov(K,CSigma-C)

    Y = np.zeros((len(t_range)))

    # Iteration through all times
    for kt, t in enumerate(t_range):
        Z = np.reshape((solSigma.y-solSS.y)[:,kt],(N,N))
        Y[kt] = np.real_if_close(solve_continuous_lyapunov(K,Z)[-1,-1])

    # Get noise
    D = (fA-fB)*C[-1,-1] + fB - 2*J*(X[-1,-1] - Y)

    # Return
    return np.real(D), np.real(J)

def g2(g,t_range):
    """
    Return master equation expression g2
    (tr[J exp(Lt) J rho] - J^2)/J^2

    Input:  g (couplings), t_range, s (entropy)
    """
    # Setup (note, fL = fA = 1-fR, fR = fB = 1-fA)
    N, fA, fB, Pi1, PiN, K, F = getGaussianSetup(g,s=np.inf)

    # Get SS
    C = getSS_covariance(g,s=np.inf)

    # Get current
    J = C[-1,-1] - fB

    # Get CSigma
    CSigma = 1/J * (fA * (C * C[-1,-1] - C@PiN@C) 
                    - fB * ((np.eye(N)-C)@PiN@(np.eye(N)-C) + C*(1-C[-1,-1])))

    # ODE
    def func(t,y):
        return np.ravel(K@np.reshape(y,(N,N)) + np.reshape(y,(N,N))@(K.conj().T))

    solSigma = solve_ivp(func,(min(t_range),max(t_range)),np.ravel(CSigma),t_eval=t_range)
    solSS    = solve_ivp(func,(min(t_range),max(t_range)),np.ravel(C),t_eval=t_range)

    # Solution
    Y = np.zeros((len(t_range)))

    # Iteration through all times
    for kt, t in enumerate(t_range):
        Z = np.reshape((solSigma.y-solSS.y)[:,kt],(N,N))
        Y[kt] = np.real(Z[-1,-1]/J)+1

    # Return G2 and J
    return np.real(Y), np.real(J)



def bulk_current_COV_finite(g,t_range,k):
    """
    Return master equation expression for current and noise
    D = 2 * int dt (tr[j exp(Lt) j rho] - j^2)
    J = tr[J rho]

    where (!!!) here: j = bulk current

    Input:  g (couplings), t_range, k (offset >= 1)
    """
    # Setup (note, fL = fA = 1-fR, fR = fB = 1-fA)
    N, fA, fB, Pi1, PiN, K, F = getGaussianSetup(g,s=np.inf)

    # Bulk coupling
    gBulk = g[-k]

    # Set current operator
    J = np.zeros((N,N),dtype=np.complex128)
    # J[N//2+1,N//2] = -1j * gBulk
    # J[N//2,N//2+1] = +1j * gBulk
    J[-k,-k-1] = -1j * gBulk
    J[-k-1,-k] = +1j * gBulk

    # Get SS
    C = getSS_covariance(g,s=np.inf)

    # Get current
    _J = np.trace(J@C)

    # Get CSigma
    CSigma = 1/_J * (C@J - C@J@C + C*_J)

    # ODE
    def func(t,y):
        return np.ravel(K@np.reshape(y,(N,N)) + np.reshape(y,(N,N))@(K.conj().T))

    solSigma = solve_ivp(func,(min(t_range),max(t_range)),np.ravel(CSigma),t_eval=t_range)
    solSS    = solve_ivp(func,(min(t_range),max(t_range)),np.ravel(C),t_eval=t_range)

        
    # Get X: KX + XK^dag = CSigma - Css
    X = solve_continuous_lyapunov(K,CSigma-C)

    Y = np.zeros((len(t_range),N,N),dtype=np.complex128)

    # Iteration through all times
    for kt, t in enumerate(t_range):
        Z = np.reshape((solSigma.y-solSS.y)[:,kt],(N,N))
        Y[kt,:,:] = np.real_if_close(solve_continuous_lyapunov(K,Z))

    # Get noise
    D = -2*_J * (np.trace(J@X) - np.trace(J@Y,axis1=1,axis2=2)) #+ np.trace(J@C@(np.eye(N)-C)@J)

    # Return
    return np.real(D), np.real(_J)

def bulk_current_COV(g,k):
    """
    Return master equation expression for current and noise
    D = 2 * int dt (tr[j exp(Lt) j rho] - j^2)
    J = tr[J rho]

    where (!!!) here: j = bulk current

    Input:  g (couplings), t_range, k (offset >= 1)
    """
    # Setup (note, fL = fA = 1-fR, fR = fB = 1-fA)
    N, fA, fB, Pi1, PiN, K, F = getGaussianSetup(g,s=np.inf)

    # Bulk coupling
    gBulk = g[-k]

    # Set current operator
    J = np.zeros((N,N),dtype=np.complex128)
    # J[N//2+1,N//2] = -1j * gBulk
    # J[N//2,N//2+1] = +1j * gBulk
    J[-k,-k-1] = -1j * gBulk
    J[-k-1,-k] = +1j * gBulk

    # Get SS
    C = getSS_covariance(g,s=np.inf)

    # Get current
    _J = np.trace(J@C)

    # Get CSigma
    CSigma = 1/_J * (C@J - C@J@C + C*_J)
        
    # Get X: KX + XK^dag = CSigma - Css
    X = solve_continuous_lyapunov(K,CSigma-C)

    # Get noise
    D = -2*_J * (np.trace(J@X)) 

    # Return
    return np.real(D), np.real(_J)