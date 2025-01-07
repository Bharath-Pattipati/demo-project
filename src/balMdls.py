"""
Balanced Models for Control
Benchmark Examples for Model Reduction, https://www.slicot.org/20-site/126-benchmark-examples-for-model-reduction
"""

# %% Importing libraries
import os
from mat4py import loadmat
import numpy as np

from control.matlab import lsim, ss, impulse
from scipy.linalg import fractional_matrix_power


# %% ERA and OKID Function Definitions
def ERA(YY, m, n, nin, nout, r):
    Dr = np.zeros((nout, nin))
    Y = np.zeros((nout, nin, YY.shape[2] - 1))
    for i in range(nout):
        for j in range(nin):
            Dr[i, j] = YY[i, j, 0]
            Y[i, j, :] = YY[i, j, 1:]

    assert len(Y[:, 0, 0]) == nout
    assert len(Y[0, :, 0]) == nin
    assert len(Y[0, 0, :]) >= m + n

    H = np.zeros((nout * m, nin * n))
    H2 = np.zeros((nout * m, nin * n))

    for i in range(m):
        for j in range(n):
            for Q in range(nout):
                for P in range(nin):
                    H[nout * i + Q, nin * j + P] = Y[Q, P, i + j]
                    H2[nout * i + Q, nin * j + P] = Y[Q, P, i + j + 1]

    U, S, VT = np.linalg.svd(H, full_matrices=0)
    V = VT.T
    Sigma = np.diag(S[:r])
    Ur = U[:, :r]
    Vr = V[:, :r]
    Ar = (
        fractional_matrix_power(Sigma, -0.5)
        @ Ur.T
        @ H2
        @ Vr
        @ fractional_matrix_power(Sigma, -0.5)
    )
    Br = fractional_matrix_power(Sigma, -0.5) @ Ur.T @ H[:, :nin]
    Cr = H[:nout, :] @ Vr @ fractional_matrix_power(Sigma, -0.5)
    HSVs = S

    return Ar, Br, Cr, Dr, HSVs


def OKID(y, u, r):
    # inputs:  y (sampled output), u (sampled input), r (effective system order)
    # outputs: H (Markov parameters), M (Observer gain)

    PP = y.shape[0]  # number of outputs
    MM = y.shape[1]  # number of output samples
    QQ = u.shape[0]  # number of inputs
    lu = u.shape[1]  # number of input samples

    assert MM == lu

    LL = r * 5

    # Form data matrices y and V
    V = np.zeros((QQ + (QQ + PP) * LL, MM))
    for i in range(MM):
        V[:QQ, i] = u[:QQ, i]

    for i in range(1, LL + 1):
        for j in range(MM - i):
            vtemp = np.concatenate((u[:, j], y[:, j]))
            V[QQ + (i - 1) * (QQ + PP) : QQ + i * (QQ + PP), i + j] = vtemp

    # Solve for observer Markov parameters Ybar
    Ybar = y @ np.linalg.pinv(V, rcond=10 ** (-3))

    # Isolate system Markov parameters H, and observer gain M
    D = Ybar[:, :QQ]  # feed-through term (or D matrix) is the first term

    Y = np.zeros((PP, QQ, LL))
    Ybar1 = np.zeros((PP, QQ, LL))
    Ybar2 = np.zeros((PP, QQ, LL))

    for i in range(LL):
        Ybar1[:, :, i] = Ybar[:, QQ + (QQ + PP) * i : QQ + (QQ + PP) * i + QQ]
        Ybar2[:, :, i] = Ybar[:, QQ + (QQ + PP) * i + QQ : QQ + (QQ + PP) * (i + 1)]

    Y[:, :, 0] = Ybar1[:, :, 0] + Ybar2[:, :, 0] @ D
    for k in range(1, LL):
        Y[:, :, k] = Ybar1[:, :, k] + Ybar2[:, :, k] @ D
        for i in range(k - 1):
            Y[:, :, k] += Ybar2[:, :, i] @ Y[:, :, k - i - 1]

    H = np.zeros((D.shape[0], D.shape[1], LL + 1))
    H[:, :, 0] = D

    for k in range(1, LL + 1):
        H[:, :, k] = Y[:, :, k - 1]

    return H


# %% Import Benchmark Examples
# example of a transmission line model [order=256, inputs = 2, outputs = 2, D = 0]
# mat_data = loadmat(os.path.join("..", "data", "external", "tline.mat"))

# Model of an atmospheric storm tracker [order=598, inputs = 1, outputs = 1, D = 0]
# mat_data = loadmat(os.path.join("..", "data", "external", "eady.mat"))

# %% System matrices and model size
""" print(mat_data.keys())
A = mat_data["A"]
B = mat_data["B"]
C = mat_data["C"]
D = 0
sysFull = ss(A, B, C, D)
q = 1
p = 1 """

q = 2  # Number of inputs
p = 2  # Number of outputs
n = 100  # State dimension
r = 10  # Reduced model order

testSys_mat = loadmat(os.path.join("..", "data", "external", "testSys_ABCD.mat"))
A = testSys_mat["A"]
B = testSys_mat["B"]
C = testSys_mat["C"]
D = testSys_mat["D"]

sysFull = ss(A, B, C, D, 1)
yFull = np.zeros((r * 5 + 2, p, q))
tspan = np.arange(0, (r * 5 + 2), 1)
m = len(tspan)

for qi in range(q):
    yFull[:, :, qi], t = impulse(sysFull, T=tspan, input=qi)


YY = np.transpose(yFull, axes=(1, 2, 0))  # reorder to size p x q x m
# %% Balanced Realization (needs slycot library)
""" A = np.array([[-0.75, 1], [-0.3, -0.75]])
B = np.array([2, 1]).reshape((2, 1))
C = np.array([1, 2])
D = 0
sys = ss(A, B, C, D)
Wc = gram(sys, "c")  # Controllability Gramian, need slycot for this
Wo = gram(sys, "o")  # Observability Gramian, need slycot for this
sysb = balred(sys, len(B))  # Balance the system, need slycot for this
BWc = gram(sysb, "c")  # Balanced Gramians
BWo = gram(sysb, "o") """

# %% Discrete Random System (needs slycot library)
""" q = 2  # Number of inputs
p = 2  # Number of outputs
r = 10  # Truncation order
n = 100  # State dimension
sysFull = drss(n, p, q)  # Discrete random system
hsvs = hsvd(sysFull)  # Hankel singular values
sysBT = balred(sysFull, r)  # Balanced truncation """

# %% Approximate impulse response with OKID
## Compute random input simulation for OKID
uRandom = np.random.randn(200, q)  # Random forcing input
yRandom = lsim(sysFull, uRandom, range(200))[0].T  # Output

# %% ERA Model
for qi in range(q):
    yFull[:, :, qi], t = impulse(sysFull, T=tspan, input=qi)
YY = np.transpose(yFull, axes=(1, 2, 0))  # reorder to p x q x m

## Compute ERA from impulse response
mco = int(np.floor((yFull.shape[0] - 1) / 2))  # m_c=m_o=(m-1)/2
Ar, Br, Cr, Dr, HSVs = ERA(YY, mco, mco, q, p, r)
sysERA = ss(Ar, Br, Cr, Dr, 1)

# %% Compute OKID and then ERA
H = OKID(yRandom, uRandom, r)
mco = int(np.floor((H.shape[2] - 1) / 2))  # m_c = m_o
Ar, Br, Cr, Dr, HSVs = ERA(H, mco, mco, q, p, r)
sysERAOKID = ss(Ar, Br, Cr, Dr, 1)

# %% Impulse Response of various models
for qi in range(q):
    y1[:, :, qi], t1 = impulse(sysFull, np.arange(200), input=qi)
    y2[:, :, qi], t2 = impulse(sysERA, np.arange(100), input=qi)
    y3[:, :, qi], t3 = impulse(sysERAOKID, np.arange(100), input=qi)
