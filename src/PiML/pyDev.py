"""
Examples from book "Data-Driven Science and Engineering" by Steven L. Brunton and J. Nathan Kutz
"""

# %% import libraries
import numpy as np
# from scipy.optimize import minimize

# import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# %% Heat equation RHS
""" def rhsHeat(uhat_ri, t, kappa, a):
    uhat = uhat_ri[:N] + (1j) * uhat_ri[N:]
    d_uhat = -(a**2) * (np.power(kappa, 2)) * uhat
    d_uhat_ri = np.concatenate((d_uhat.real, d_uhat.imag)).astype("float64")
    return d_uhat_ri """

# %% print version
""" print(np.__version__)
print(pd.__version__) """

# %% Dataframe
""" data = {
    "Name": ["John", "Anna", "Peter", "Linda"],
    "Age": [23, 36, 28, 45],
    "City": ["New York", "Paris", "Berlin", "London"],
}

df = pd.DataFrame(data)
df = df.replace("New York", "New York City") """

# %% SVD
""" X = np.random.rand(5, 3)  # Create random data matrix
U, S, V = np.linalg.svd(X, full_matrices=True)  # Perform SVD
Uhat, Shat, VThat = np.linalg.svd(X, full_matrices=False)  # Economy SVD """

# %% Least Squares fit of noisy data
""" x = 3  # True slope
a = np.arange(-2, 2, 0.25)
a = a.reshape(-1, 1)
b = x * a + np.random.randn(*a.shape)  # Add noise
plt.plot(a, x * a, color="k", linewidth=2, label="True Line")  # True relationship
plt.plot(a, b, "x", color="r", markersize=10, label="Noisydata")  # Noisy measurements

# Compute least-squares approximation with the SVD
U, S, VT = np.linalg.svd(a, full_matrices=False)  # Economy SVD
xtilde = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ b  # Least-squares fit
plt.plot(a, xtilde * a, "--", color="b", linewidth=2, label="Regression line")

plt.ylabel(r"b $\rightarrow$")
plt.xlabel(r"a $\rightarrow$")
plt.legend()

# Alternative formulations of least squares
xtilde1 = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ b
xtilde2 = np.linalg.pinv(a) @ b """

# %% PCA example on noisy cloud data
""" 
# Generate noisy cloud of data
xC = np.array([2, 1])  # Center of data (mean)
sig = np.array([2, 0.5])  # Principal axes
theta = np.pi / 3  # Rotate cloud by pi/3
R = np.array(
    [
        [np.cos(theta), -np.sin(theta)],  # Rotation mat
        [np.sin(theta), np.cos(theta)],
    ]
)

nPoints = 10000  # Create 10,000 points
X = R @ np.diag(sig) @ np.random.randn(2, nPoints) + np.diag(xC) @ np.ones((2, nPoints))

f1 = plt.figure()
ax1 = f1.add_subplot(121)
ax2 = f1.add_subplot(122)
ax1.plot(X[0, :], X[1, :], "o", color="k")  # Plot data
Xavg = np.mean(X, axis=1)  # Compute mean
B = X - np.tile(Xavg, (nPoints, 1)).T  # Mean-subtracted data

# Find principal components (SVD)
U, S, VT = np.linalg.svd(B / np.sqrt(nPoints), full_matrices=0)
theta = 2 * np.pi * np.arange(0, 1, 0.01)
Xstd = (
    U @ np.diag(S) @ np.array([np.cos(theta), np.sin(theta)])
)  # Matrix Multiplication

# First three standard deviation ellipsoids
ax2.plot(X[0, :], X[1, :], "o", color="b")  # Plot data
ax2.plot(Xavg[0] + Xstd[0, :], Xavg[1] + Xstd[1, :], "-", color="r", linewidth=3)
ax2.plot(
    Xavg[0] + 2 * Xstd[0, :], Xavg[1] + 2 * Xstd[1, :], "-", color="r", linewidth=3
)
ax2.plot(
    Xavg[0] + 3 * Xstd[0, :], Xavg[1] + 3 * Xstd[1, :], "-", color="r", linewidth=3
)
plt.show() """

# %% Compare various thresholding approaches on noisy low-rank data
""" 
# Generate underlying low-rank data
t = np.arange(-3, 3, 0.01)
Utrue = np.array([np.cos(17 * t) * np.exp(-(t**2)), np.sin(11 * t)]).T
Strue = np.array([[2, 0], [0, 0.5]])
Vtrue = np.array([np.sin(5 * t) * np.exp(-(t**2)), np.cos(13 * t)]).T
X = Utrue @ Strue @ Vtrue.T

f1 = plt.figure()
ax1 = f1.add_subplot(221)
ax2 = f1.add_subplot(222)
ax3 = f1.add_subplot(223)
ax4 = f1.add_subplot(224)
ax1.imshow(X)

# Contaminate signal with noise
sigma = 1
Xnoisy = X + sigma * np.random.randn(*X.shape)
ax2.imshow(Xnoisy)

# Truncate using optimal hard threshold
U, S, VT = np.linalg.svd(Xnoisy, full_matrices=0)
N = Xnoisy.shape[0]
cutoff = (4 / np.sqrt(3)) * np.sqrt(N) * sigma  # Hard threshold
r = np.max(np.where(S > cutoff))  # Keep modes w/ S > cutoff
Xclean = U[:, : (r + 1)] @ np.diag(S[: (r + 1)]) @ VT[: (r + 1), :]
ax3.imshow(Xclean)

# Truncate to keep 90% of cumulative sum
cdS = np.cumsum(S) / np.sum(S)  # Cumulative energy
r90 = np.min(np.where(cdS > 0.90))  # Find r to keep 90% sum

X90 = U[:, : (r90 + 1)] @ np.diag(S[: (r90 + 1)]) @ VT[: (r90 + 1), :]
ax4.imshow(X90)
plt.show()
 """

# %% 1D heat equation using Fourier transform
""" a = 1  # Thermal diffusivity constant
L = 100  # Length of domain
N = 1000  # Number of discretization points
dx = L / N
x = np.arange(-L / 2, L / 2, dx)  # Define x domain

# Define discrete wavenumbers
kappa = 2 * np.pi * np.fft.fftfreq(N, d=dx)

# Initial condition
u0 = np.zeros_like(x)
u0[int((L / 2 - L / 10) / dx) : int((L / 2 + L / 10) / dx)] = 1
u0hat = np.fft.fft(u0)

# Simulate in Fourier frequency domain
dt = 0.1
t = np.arange(0, 10, dt)
u0hat_ri = np.concatenate((u0hat.real, u0hat.imag)).astype("float64")
uhat_ri = odeint(rhsHeat, u0hat_ri, t, args=(kappa, a))
uhat = uhat_ri[:, :N] + (1j) * uhat_ri[:, N:]
u = np.zeros_like(uhat)
for k in range(len(t)):
    u[k, :] = np.fft.ifft(uhat[k, :])
u = u.real

# waterfall plot
X, T = np.meshgrid(x, t)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, T, u, cmap="viridis")

ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u")
plt.show()

# X-T Contour plot
plt.figure()
plt.contourf(X, T, u, 20, cmap="inferno")
plt.xlabel("x")
plt.ylabel("t")
plt.colorbar(label="u")
plt.title("X-T Diagram")
plt.show() """

# %% Spectrogram of quadractic chirp
""" dt = 0.001
t = np.arange(0, 2, dt)
f0 = 50
f1 = 250
t1 = 2
x = np.cos(2 * np.pi * t * (f0 + (f1 - f0) * np.power(t, 2) / (3 * t1**2)))
plt.specgram(x, NFFT=128, Fs=1 / dt, noverlap=120, cmap="jet")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Spectrogram of Quadratic Chirp")
plt.show() """

# %% L1-norm for robust statical regression
""" x = np.sort(4 * (np.random.rand(25, 1) - 0.5), axis=0)  # data
b = 0.9 * x + 0.1 * np.random.randn(len(x), 1)  # Noisy line y=ax
atrue = np.linalg.lstsq(x, b, rcond=None)[0]  # Least-squares a
b[-1] = -5.5  # Introduce outlier
acorrupt = np.linalg.lstsq(x, b, rcond=None)[0]  # New slope

plt.plot(x, b, "o", label="Data")
plt.plot(x, x * atrue, "k", label="True LS fit")
plt.plot(x, x * acorrupt, "r", label="L2-norm fit")
plt.legend()


## L1 optimization to reject outlier
def L1_norm(a):
    return np.linalg.norm(a * x - b, ord=1)


a0 = acorrupt.flatten()  # initialize to L2 solution
res = minimize(L1_norm, a0)
aL1 = res.x[0]  # aL1 is robust
plt.plot(x, x * aL1, "g--", label="L1-norm fit")
plt.legend()
plt.show() """


# %% Data-driven dynamical systems
# Lorenz system
def lorenz(x_y_z, t0, sigma=10, beta=8 / 3, rho=28):
    x, y, z = x_y_z
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


beta = 8 / 3
sigma = 10
rho = 28
x0 = (0, 1, 20)
dt = 0.001
t = np.arange(0, 50 + dt, dt)
x_t = odeint(lorenz, x0, t, rtol=10 ** (-12), atol=10 ** (-12) * np.ones_like(x0))
x, y, z = x_t.T

# 3D plot of x, y, z
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(x, y, z, linewidth=1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.title("Lorenz Attractor")
plt.show()

# %% HAVOK (Hankel Alternaltive View of Koopman) Code for Lorenz System

## Eigen-time delay coordinates
stackmax = 10  # Number of shift-stacked rows, q from equation (7.107)
r = 10  # rank of HAVOK model
H = np.zeros((stackmax, x_t.shape[0] - stackmax))  # Time-shifted matrix initialization

for k in range(stackmax):
    H[k, :] = x_t[
        k : -(stackmax - k), 0
    ]  # Hankel matrix from a single measurement x(t)
U, S, VT = np.linalg.svd(H, full_matrices=0)
V = VT.T

## Compute Derivatives (4th Order Central Difference)
dV = (1 / (12 * dt)) * (-V[4:, :] + 8 * V[3:-1, :] - 8 * V[1:-3, :] + V[:-4, :])

# trim first and last two that are lost in derivative
V = V[2:-2]

## Build HAVOK Regression Model on Time Delay Coordinates
Xi = np.linalg.lstsq(V, dV, rcond=None)[0]

# Refer to equation (7.109)
# Linear model on first (r-1) variables and recast last variable as a forcing term
A = Xi[: (r - 1), : (r - 1)].T
B = Xi[-1, : (r - 1)].T

# %%
