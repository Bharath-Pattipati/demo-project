# %% Import libraries
import numpy as np

from scipy.optimize import minimize
import matplotlib.pyplot as plt
# import time
# from scipy.fftpack import dct, idct

# CoSaMP: https://github.com/rfmiotto/CoSaMP
# Original Needell and Tropp 2008 paper: https://arxiv.org/abs/0803.2392
# from cosamp.cosamp import cosamp

# %% Sparse solutions to under-determined linear system.
# Under-determined linear system: y = Theta * s i.e. more unknowns than knowns
""" 
tic = time.time()
# Solve y = Theta * s for "s"
n = 1000  # dimension of s
p = 200  # number of measurements, dim(y)
Theta = np.random.randn(p, n)
y = np.random.randn(p)


# L1 Minimum norm solution s_L1
def L1_norm(x):
    return np.linalg.norm(x, ord=1)


constr = {"type": "eq", "fun": lambda x: Theta @ x - y}
x0 = np.linalg.pinv(Theta) @ y  # initialize with L2 solution
res = minimize(L1_norm, x0, method="SLSQP", constraints=constr)
s_L1 = res.x

toc = time.time()  # End timer
elapsed_time = toc - tic
print(f"Elapsed time: {elapsed_time:.4f} seconds")

# Create a line plot
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

# Line plot in the first subplot
ax[0].plot(s_L1)
ax[0].set_xlabel("Index")
ax[0].set_ylabel(r"$s_j$")
ax[0].set_title("Under-determined Linear System")

# Histogram in the second subplot
ax[1].hist(s_L1, bins=30)
ax[1].set_xlabel("Value")
ax[1].set_ylabel(r"Hist$(s_j)$")
ax[1].set_title("Histogram of sparse solution")

plt.tight_layout()
plt.show() """

# %% Recovering an Audio Signal from Sparse Measurements
# Compressed sensing of two-tone cosine signal.

""" tic = time.time()
## Generate signal, DCT of signal
n = 4096  # points in high resolution signal
f1 = 100  # first-frequency (Hz)
f2 = 500  # second-frequency (Hz)
t = np.linspace(0, 1, n)
x = np.cos(2 * f1 * np.pi * t) + np.cos(2 * f2 * np.pi * t)  # Two-tone signal
xt = np.fft.fft(x)  # Fourier transformed signal
PSD = xt * np.conj(xt) / n  # Power spectral density

## Randomly sample signal
# precise timing of the sparse measurements at a much higher resolution than
# our sampling rate
p = 128  # num. random samples, p = n/32
perm = np.floor(np.random.rand(p) * n).astype(int)
y = x[perm]

## Solve compressed sensing problem
Psi = dct(np.identity(n))  # Build discrete cosine transform basis
Theta = Psi[perm, :]  # Measure rows of Psi

# CS via matching pursuit
s = cosamp(Theta, y, 10, tol=1e-10, max_iter=100)
xrecon = idct(s)  # reconstruct full signal
xr = np.fft.fft(x)  # Fourier transformed signal
PSDr = xr * np.conj(xr) / n  # Power spectral density

toc = time.time()  # End timer
elapsed_time = toc - tic
print(f"Elapsed time: {elapsed_time:.4f} seconds")

# Create a line plot
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

# Line plot in the first subplot
ax[0].plot(x, label="Original Signal", color="blue")
ax[0].plot(xrecon, label="Reconstructed Signal", color="red", linestyle="--")
ax[0].legend()
ax[0].set_xlabel("Time")
ax[0].set_ylabel(r"X")
ax[0].set_title("Compressed Sensing of Two-Tone Signal")

ax[1].plot(PSD, label="Original Signal", color="blue")
ax[1].plot(PSDr, label="Reconstructed Signal", color="red", linestyle="--")
ax[1].legend()
ax[1].set_xlabel("Frequency")
ax[1].set_ylabel(r"PSD")
ax[1].set_xlim(0, 1000)
ax[1].set_title("Power Spectral Density")

plt.tight_layout()
plt.show() """


# %% Function Visualization
""" def func2D(x, y):
    return x**2 + 3 * y**2


# visualization of function along x and y
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = func2D(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Visualization of func2D")

plt.show() """

# %% Solutions for an over-determined linear system
n = 500
m = 100
A = np.random.rand(n, m)
b = np.random.rand(n)

xdag = np.linalg.pinv(A) @ b
lam = np.array([0, 0.1, 0.5])


def reg_norm(x, A, b, lam):
    return np.linalg.norm(A @ x - b, ord=2) + lam * np.linalg.norm(x, ord=1)


f1, ax1 = plt.subplots(3, 1, figsize=(10, 10))
f2, ax2 = plt.subplots(1, 3, figsize=(5, 5))
for j in range(3):
    res = minimize(reg_norm, x0=xdag, args=(A, b, lam[j]))
    x = res.x

    ax1[j].bar(np.arange(m), x)
    ax1[j].set_title(f"Regularization parameter: {lam[j]}")
    ax1[j].set_xlabel("Index")
    ax1[j].set_ylabel(r"$x_j$")

    ax2[j].hist(x, bins=20)
    ax2[j].set_title(f"$\lambda$ = {lam[j]}")
    ax2[j].set_ylabel(r"Hist$(x_j)$")
    ax2[j].set_xlim(-0.1, 0.1)

plt.tight_layout()
plt.show()

# %%
