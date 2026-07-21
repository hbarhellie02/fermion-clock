import numpy as np
from time import time
from numba import njit, prange, get_num_threads, set_num_threads

import sys
print(sys.version)

# Access the first argument after the script name
try:
    lbl = sys.argv[1]
    print(f"A script label has been passed: {lbl}")
    lbl = "_"+lbl
except:
    lbl = ""

@njit
def check_nan(name, array):
    if np.any(np.isnan(array)):
        print(f"Warning! {name} contains NaNs")
        return True
    return False

@njit
def weighted_choice(cases, probabilities):
    cum_probs = np.cumsum(probabilities)
    r = np.random.rand()
    for i in range(cum_probs.shape[0]):
        if r < cum_probs[i]:
            return cases[i]
    return cases[-1]  # Fallback in case of rounding errors

@njit
def dF(C,H,Pi1,PiN):
    """
    Returns derivative of the no-jump evolution of the covariance matrix
    Input:
        C   :   covariance matrix
        H   :   Hamiltonian
        Pi1 :   Projector on 1st mode
        PiN :   Projector on Nth mode
    Returns:
        derivative at C
    """
    # Hermitian evolution
    C_hermitian = +1j * (H.dot(C) - C.dot(H))

    # Jump steps
    C_jump1 =  - 1/2 * ((PiN - Pi1).dot(C) + C.dot(PiN - Pi1))
    C_jump2 =  + C.dot((PiN - Pi1).dot(C)) #- (C[-1,-1] + 1 - C[0,0]) * C

    # Final Covariance matrix return
    return C_hermitian + C_jump1 + C_jump2

@njit
def dF_finite(C,H,Pi1,PiN,s):
    """
    Returns derivative of the no-jump evolution of the covariance matrix
    Input:
        C   :   covariance matrix
        H   :   Hamiltonian
        Pi1 :   Projector on 1st mode
        PiN :   Projector on Nth mode
        s   :   entropy
    Returns:
        derivative at C
    """
    # Hermitian evolution
    C_hermitian = +1j * (H.dot(C) - C.dot(H))

    # Jump steps
    C_jump1 =  - 1/2 * ((PiN - Pi1).dot(C) + C.dot(PiN - Pi1))
    C_jump2 =  + C.dot((PiN - Pi1).dot(C)) #- (C[-1,-1] + 1 - C[0,0]) * C

    # Final Covariance matrix return
    return C_hermitian + (1-np.exp(-s))*(C_jump1 + C_jump2)

@njit
def jump_probabilities(C,eps):
    """
    Return the probabilities that a jump is made
    Input:
        C   :   covariance matrix
        eps :   step size
    Return:
        P   :   [P0, PL, PR] jump probabilities
    """
    # Return probabilties
    return np.abs(np.array([1 - eps * (1 - C[0,0]) - eps * C[-1,-1],eps * (1 - C[0,0]),eps * C[-1,-1]]))

@njit
def jump_probabilities_finite(C,eps,s):
    """
    Return the probabilities that a jump is made
    Input:
        C   :   covariance matrix
        eps :   step size
        s   :   entropy
    Return:
        P   :   [P0, PLin, PRout, PLout, PRin] jump probabilities
    """
    # Return probabilties
    return np.abs(np.array([1 - eps * (1 - C[0,0]) - eps * C[-1,-1] - eps * np.exp(-s) * C[0,0] - eps * np.exp(-s) * (1 - C[-1,-1]),
                            eps * (1 - C[0,0]),
                            eps * C[-1,-1],
                            eps * np.exp(-s) * C[0,0],
                            eps * np.exp(-s) * (1 - C[-1,-1])]))

@njit
def RK_step(C,eps,H,Pi1,PiN,s=0):
    """
    Return a RK step evolution following the RK4 tableau; note that this
    function specifically only works for dF defined in this file
    Input:
        C       :   initial covariance matrix
        eps     :   time step size
        H       :   Hamiltonian
        Pi1/PiN :   projectors needed for derivative
        s       :   s=0 is standard, if >0 then finite temperature
    Return:
        C1  :   new covariance matrix
    """
    if s==0:
        # Find steps
        k1 = dF(C,H,Pi1,PiN)
        k2 = dF(C + eps * 0.5 * k1,H,Pi1,PiN)
        k3 = dF(C + eps * 0.5 * k2,H,Pi1,PiN)
        k4 = dF(C + eps * k3,H,Pi1,PiN)
    else:
        # Find steps
        k1 = dF_finite(C,H,Pi1,PiN,s)
        k2 = dF_finite(C + eps * 0.5 * k1,H,Pi1,PiN,s)
        k3 = dF_finite(C + eps * 0.5 * k2,H,Pi1,PiN,s)
        k4 = dF_finite(C + eps * k3,H,Pi1,PiN,s)

    # Summation
    C1 = C + eps * (k1/6 + k2/3 + k3/3 + k4/6)

    # Return statement
    return C1

@njit
def svd_correction(C,M):
    """
    Correction step where covariance matrix is projected onto the correct
    Slater determinant space
    Input:
        C   :   cov matrix
        M   :   number of particles
    Return:
        C1  :   corrected cov matrix
    """
    # Dimensionality
    n = C.shape[0]

    # Still need to regularize using SVD
    try:
        U, S, Vh = np.linalg.svd(C)
    except Exception:
        print("Poop, caught a linalg error")
        U, S, Vh = np.linalg.svd(C+1e-5*np.eye(n))

    # U, S, Vh = np.linalg.svd(C1,hermitian=True)
    
    S1 = np.zeros((n),dtype=np.complex128)
    S1[:M] = 1.0

    # Force positivity
    C1 = U.dot(np.diag(S1).dot(U.T.conj()))

    # Check stability
    NaN1 = check_nan("C1",C1)
    if NaN1:
        print("... attempting regularization.")
        # Force regularization
        U, S, Vh = np.linalg.svd(C+1e-5*np.eye(n))
        C2 = U.dot(np.diag(S1).dot(U.T.conj()))
        NaN2 = check_nan("C2",C2)
        if NaN2:
            raise Exception("Second iteration failed")
        return C2

    return C1

@njit
def step_jumpL(C):
    """
    Return normalized state after jump to the left.
    Input:
        C   :   initial covariance matrix
    Return:
        C1  :   state after jump, normalized
    """
    # Obtain dimension
    n = C.shape[0]

    # Useful projection matrices
    Pi1 = np.zeros((n),dtype=np.complex128)
    Pi1[0] = 1.0
    Pi1 = np.diag(Pi1)

    # Prepare return state
    C1 = (np.eye(n) - C).dot(Pi1).dot(np.eye(n) - C) + (1-C[0,0])*C
    C1 = C1 / (1 - C[0,0])

    return C1

@njit
def step_jumpL_out(C):
    """
    Return normalized state after jump to the left --> leaving the system
    Input:
        C   :   initial covariance matrix
    Return:
        C1  :   state after jump, normalized
    """
    # Obtain dimension
    n = C.shape[0]

    # Useful projection matrices
    Pi1 = np.zeros((n),dtype=np.complex128)
    Pi1[0] = 1.0
    Pi1 = np.diag(Pi1)

    # Prepare return state
    C1 = C - C.dot(Pi1.dot(C)) / C[0,0]

    return C1

@njit
def step_jumpR(C):
    """
    Return normalized state after jump to the right.
    Input:
        C   :   initial covariance matrix
    Return:
        C1  :   state after jump, normalized
    """
    # Obtain dimension
    n = C.shape[0]

    # Useful projection matrices
    PiN = np.zeros((n),dtype=np.complex128)
    PiN[-1] = 1.0
    PiN = np.diag(PiN)

    # Prepare return state
    C1 = (C[-1,-1] * C - C.dot(PiN.dot(C))) / C[-1,-1]

    # Fix signs from Wigner-String
    W = np.diag((-1)**np.diag(PiN))
    C1 = W.dot(C1.dot(W))

    return C1

@njit
def step_jumpR_in(C):
    """
    Return normalized state after jump on the right --> entering the system.
    Input:
        C   :   initial covariance matrix
    Return:
        C1  :   state after jump, normalized
    """
    # Obtain dimension
    n = C.shape[0]

    # Useful projection matrices
    PiN = np.zeros((n),dtype=np.complex128)
    PiN[-1] = 1.0
    PiN = np.diag(PiN)

    # Prepare return state
    C1 = C + (np.eye(n)-C).dot(PiN.dot(np.eye(n)-C)) / (1 - C[-1,-1])

    # Fix signs from Wigner-String
    W = np.diag((-1)**np.diag(PiN))
    C1 = W.dot(C1.dot(W))

    return C1

@njit(parallel=True)
def mc_evolution(H,eps,Ntick=1000,Nsamp=10,s=0,n_skip=3,preCompile=False):
    """
    Stochastic trajectory simulation
    Input:
        H   :   Quadratic fermionic Hamiltonian
        tf  :   final time of evolution
        eps :   step size
      Ntick :   Number of ticks on RHS (note that there will be 2*n ticks padding on the left)
      Nsamp :   Number of samples
        s   :   entropy

    Return:
        times_L :   jump times left (np array)
        times_R :   jump times right (np array)
    """
    print("mc_evolution call successful!")

    # Identify Problem size
    n = H.shape[0]

    # Prepare list of tick times
    Ntick_true = Ntick + 2*n
    times_right = np.zeros((Nsamp,Ntick_true))

    # Useful projection matrices
    Pi1 = np.zeros((n),dtype=np.complex128)
    Pi1[0] = 1.0
    Pi1 = np.diag(Pi1)

    PiN = np.zeros((n),dtype=np.complex128)
    PiN[-1] = 1.0
    PiN = np.diag(PiN)

    # Status of threads
    status = np.zeros((Nsamp),dtype=np.integer)

    # Parallelizable for loop
    for k_samp in prange(Nsamp):
        # Local times
        times_local = np.zeros((Ntick_true))

        # Initial state
        C = np.zeros((n,n),dtype=np.complex128)
        C[0,0] = 1.0

        # Initial number of excitations and jump
        M = 1
        Mr = 0

        # Iteration
        t_it = 0

        # SVD skipper
        svd_skipper = 0

        while Mr < Ntick_true:
            # Increment iteration time
            t_it += eps
            
            # Determine jump probabilities and which jump
            if s==0:
                P = jump_probabilities(C,eps)
                # Decide what event
                cases = np.array([0,1,2])
                case = weighted_choice(cases,P)
            else:
                P = jump_probabilities_finite(C,eps,s)
                # Decide what event, 0, Lin, Rout, Lout, Rin
                cases = np.array([0,1,2,3,4])
                case = weighted_choice(cases,P)

            # No jump evolution
            if case == 0:
                if svd_skipper==0:
                    # Evolution step
                    C = svd_correction(RK_step(C,eps,H,Pi1,PiN,s=s),M)
                else:
                    # Evolution step
                    C = RK_step(C,eps,H,Pi1,PiN,s=s)

            # Jump Left IN
            elif case == 1:
                C = step_jumpL(C)
                M+=1

            # Jump Right OUT
            elif case == 2:
                C = step_jumpR(C)
                times_local[Mr] = t_it
                # times_right[k_samp,Mr] = t_it
                Mr+=1
                M-=1
                # Set end condition
                if Mr==Ntick_true:
                    status[k_samp] = 1

                # Print status update
                if (Mr)%100==0:
                    print("> Thread "+str(k_samp)+"; Mr = "+str(Mr)+" / "+str(Ntick_true))

            # Jump Left OUT
            elif case == 3:
                C = step_jumpL_out(C)
                M-=1

            # Jump Right IN
            elif case == 4:
                C = step_jumpR_in(C)
                M+=1
                # Correction of the forwards tick 
                # Note: the time will be overwritten
                Mr-=1

            if preCompile:
                break

            if n_skip>=1:
                svd_skipper=(svd_skipper+1)%n_skip

        times_right[k_samp,:] = times_local
        print("Success! Thread "+str(k_samp)+" has just ended.")

    # Return
    return times_right[:,(2*n - 1):]

def num_exp_temperatures(N=20,s_list=np.array([4.,6.,8.,10.]),Nsamp = 8,Ntick = 20005):
    ##########################
    # Systematic simulation  #
    # for finite temperature #
    ##########################

    # Initial setup
    g = np.load("params/g_n"+str(N)+".npy")
    H = np.diag(g,-1) + np.diag(g,+1)
    H = H.astype(np.complex128)

    # Numba threads
    default_n_thread = get_num_threads()
    if default_n_thread>Nsamp:
        set_num_threads(Nsamp)

    # Numba status
    print("Number of Numba threads:", get_num_threads())

    # Precompilation
    print("Start pre-compiling")
    start = time()
    mc_evolution(H,0.1,Ntick,Nsamp,0,preCompile=True)
    end = time()
    print(f"\nCompilation time: {end - start:.6f} seconds\n")

    print("Starting iteration for finite-temp simulation ...")

    for k_s, s in enumerate(s_list):
        # Evolution
        print("Starting case s = "+str(np.round(s,2))+" ...\n")
        start = time()
        times_parallel = mc_evolution(H,0.04,Ntick,Nsamp,s,preCompile=False)
        end = time()
        print(f"\nExecution time: {end - start:.6f} seconds\n")

        # Statistics
        np.save("data/times_finite/times_parallel_n"+str(N)+"_s"+str(np.round(s,5))+".npy",times_parallel)

    np.save("data/times_finite/s_range_n"+str(N)+"_20251008.npy",s_list)

    return None

def num_experiment_lengths(N_list = [10,20,40,60],Nsamp=8,Ntick=20005):
    ##########################
    # Systematic simulation  #
    # for T=0, many N's      #
    ##########################
    for k_N, N in enumerate(N_list):
        # Standard case
        g = np.load("params/g_n"+str(N)+".npy")
        # Perturbed Params
        # g = np.load("params_perturby/g_n"+str(N)+".npy")

        # Set Hamiltonian
        H = np.diag(g,-1) + np.diag(g,+1)
        H = H.astype(np.complex128)

        # Numba threads
        default_n_thread = get_num_threads()
        if default_n_thread>Nsamp:
            set_num_threads(Nsamp)

        # Numba status
        print("Number of Numba threads:", get_num_threads())

        if k_N==0:
            # Precompilation
            print("Start pre-compiling ...\n")
            start = time()
            mc_evolution(H,0.1,Ntick,Nsamp,preCompile=True)
            end = time()
            print(f"\nCompilation time: {end - start:.6f} seconds\n")

        # Evolution
        print("Start ...\n")
        start = time()
        times_parallel = mc_evolution(H,0.04,Ntick,Nsamp,preCompile=False)
        end = time()
        print(f"\nExecution time: {end - start:.6f} seconds\n")

        # Statistics
        np.save("data/tmp/times_parallel_n"+str(N)+".npy",times_parallel)

    return None

if __name__ == "__main__":
    num_exp_temperatures(N=20,s_list=np.array([5.0]),Nsamp=16,Ntick=20060)
    # num_experiment_lengths(N_list=[40],Nsamp=16,Ntick=1000)


# PERFORMANCE STATS with new bundle, Oct. 2025

# Perturbed params
# Threads = 16
# N = 40, 1080 ticks: 255 sec
# N = 40, 22440 sec / 100090 ticks
# N = 40, 22635 sec / 100090 ticks

# Entropy samples = 1
# Threads = 16
# N = 40,   254 sec /   1080 ticks
# N = 40, 31187 sec / 100090 ticks
# N = 40, 32589 sec / 100090 ticks (8 entropy)
# N = 40, 29021 sec / 100090 ticks (9 entropy)
# N = 40,   346 sec /   1100 ticks (5.5 entropy)
# N = 40, 29594 sec / 100100 ticks (5.5 entropy)

# N = 20, 86 sec / 1050 ticks
# N = 20, 7464 sec / 100050 ticks
# N = 20, 7542 sec / 100050 ticks
# N = 20, 5693 sec / 100050 ticks
# N = 20,  874 sec / 12050 ticks

# Entropy samples = 13
# Threads = 16
# N = 20, 1000 ticks: 764 sec


################
# LEGACY STATS #
################

# CLUSTER PERFORMANCE STATS
# N_THREADS = 8
# N = 40:   153 sec / 1080 ticks
# N = 50:   255 sec / 1100 ticks
# N = 50:   257 sec / 1100 ticks
# N = 60:   384 sec / 1120 ticks
# N = 70:   609 sec / 1140 ticks
# N = 80:   797 sec / 1160 ticks
# N = 100: 1451 sec / 1200 ticks

# N_THREADS = 16
# N = 40:   184 sec / 1080 ticks
# N = 40:   198 sec / 1080 ticks
# N = 60:   552 sec / 1120 ticks

## NEW STATISTICS WITH PACKAGE scipy-bundle/2023.11-gfbf-2023b
# with fixed cores, 16 threads
# N = 40:   141 sec / 1080 ticks
# N = 60:   363 sec / 1120 ticks
# N = 80:  1163 sec / 1160 ticks
# N = 80:   958 sec / 1160 ticks

# without fixed cores, 16 threads
# N = 40:   141 sec / 1080 ticks
# N = 60:   366 sec / 1120 ticks
# N = 80:  1269 sec / 1160 ticks

# Fixed cores, 8 threads, 8 cores
# N = 80:    774 sec / 1160 ticks (step = 0.1, as before)
# N = 80:   1119 sec / 1160 ticks (step = 0.08)
# N = 80:   1309 sec / 1160 ticks (step = 0.08)
# N = 80:   1374 sec / 1160 ticks (step = 0.04, SVD every 3rd step)
# N = 100:  1677 sec /  700 ticks (step = 0.04, SVD every 3rd step)
# N = 100:  2403 sec / 1200 ticks (step = 0.04, SVD every 3rd step)

# Fixed cores, 4 threads, 4 cores
# N = 100:  2513 sec / 1200 ticks (step = 0.04, SVD every 3rd step)

# Fixed cores, 16 threads, 16 cores
# N = 10:    434 sec / 20020 ticks (step = 0.04, SVD every 3rd step)
# N = 20:    530 sec / 10050 ticks (step = 0.04, SVD every 3rd step)
# N = 100:  3082 sec /  1200 ticks (step = 0.04, SVD every 3rd step)

# NO fixed cores, 8 threads, 8 cores
# N = 80:   804 sec / 1160 ticks

# NO fixed cores, 8 threads, 16 cores
# N = 80:   708 sec / 1160 ticks, warning: low efficiency!

### ACTUAL EXPERIMENT, 16 threads, new package bundle
# N = 40:  25916 sec / 200090 ticks
# N = 60:  64940 sec / 200130 ticks

# 8 threads, new package bundle (step = 0.04, SVD every 3rd step)
# N = 80:  206206 sec / 200170 ticks
# N = 80:  245108 sec / 200170 ticks

# 16 threads, new package bundle (step = 0.04, SVD every 3rd step)
# N = 10:    6159 sec / 200050 ticks
# N = 20:   11991 sec / 200050 ticks

### ACTUAL EXPERIMENT, 16 threads, old package bundle
# N = 60, 24929 / 50125 ticks