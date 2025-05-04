"""
DMD Class by Nathan Kutz, Dynamic Mode Decomposition Code on YouTube
This is a simple implementation of the Dynamic Mode Decomposition (DMD) algorithm.
Book: http://dmdbook.com/
"""

# %% Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# %% Define the space and time grid for data collection.
nx = 400  # number of grid points along space dimension
nt = 200  # number of grid points along time dimension
x = np.linspace(-10, 10, nx)
t = np.linspace(0, 4 * np.pi, nt)
xgrid, tgrid = np.meshgrid(x, t)
dt = t[1] - t[0]  # time step between each snapshot
r = 2  # rank of the DMD modes


# %% Create the input data by summing two different functions
def sech(x):
    return 1 / np.cosh(x)


def f1(x, t):
    return sech(x + 3) * (1.0 * np.exp(1j * 2.3 * t))


def f2(x, t):
    return sech(x) * np.tanh(x) * (2.0 * np.exp(1j * 2.8 * t))


# Combined the 2 functions. Analyze how DMD separates them.
X1 = f1(xgrid, tgrid)
X2 = f2(xgrid, tgrid)
X = X1 + X2

# %% Functions and the dataset without noise.
titles = ["$f_1(x,t)$", "$f_2(x,t)$", "$f$"]
data = [X1, X2, X]

fig = plt.figure(figsize=(17, 6), dpi=200)
for n, title, d in zip(range(131, 134), titles, data):
    plt.subplot(n)
    plt.pcolor(xgrid, tgrid, d.real)
    plt.title(title)
    plt.xlabel("Space")
    plt.ylabel("Time")
    plt.colorbar()
plt.show()

# %% Step 1: SVD (Basis functions)
U, s, V = np.linalg.svd(X.T, full_matrices=False)  # rows = spatial, cols = time

# plot singular values
plt.figure(figsize=(8, 4), dpi=200)
plt.plot(s / sum(s), "ko")
plt.title("Singular Values")
plt.xlabel("Index")
plt.ylabel("Singular Value")
plt.grid()
plt.show()

# plot left singular vectors
plt.figure(figsize=(8, 4), dpi=200)
plt.plot(U[:, 0], "ko", label="U1")
plt.plot(U[:, 1], "ro", label="U2")
plt.title("Left Singular Vector")
plt.xlabel("Index")
plt.ylabel("Spacial Modes")
plt.legend()
plt.grid()
plt.show()

# plot right singular vectors
plt.figure(figsize=(8, 4), dpi=200)
plt.plot(V[0, :], "k", label="V1")
plt.plot(V[1, :], "r", label="V2")
plt.title("Right Singular Vector")
plt.xlabel("Index")
plt.ylabel("Time Modes")
plt.legend()
plt.grid()
plt.show()

# %% DMD with perfect data (doesn't work with noisy data)
F = X.T  # Rows = spatial measurements, Columns = time measurements
F1 = F[:, :-1]
F2 = F[:, 1:]
f0 = F[:, 0]  # first column i.e. time 0

U, S, V = np.linalg.svd(F1, full_matrices=False)
Ur = U[:, :r]
Sr = S[:r]
Vr = V[:r, :]

Atilde = Ur.conj().T @ F2 @ Vr.conj().T @ np.diag(1.0 / Sr)
eigenvalues, eigenvectors = np.linalg.eig(Atilde)
phi = F2 @ Vr.conj().T @ np.diag(1.0 / Sr) @ eigenvectors
omega = np.log(eigenvalues) / dt
b = np.linalg.lstsq(phi, f0, rcond=None)[0]

# Reconstruction
time_dynamics = np.zeros((r, nt), dtype=complex)
for i in range(nt):
    time_dynamics[:, i] = b * np.exp(omega * t[i])
X_dmd = phi @ time_dynamics

# %% plot DMD modes
plt.figure(figsize=(8, 4), dpi=200)
plt.plot(U[:, 0], "ko", label="SVD1")
plt.plot(U[:, 1], "ro", label="SVD2")
plt.plot(phi.real[:, 0], label="DMD1", linewidth=3)
plt.plot(phi.real[:, 1], label="DMD2", linewidth=3)
plt.title("SVD vs. DMD Modes")
plt.xlabel("Time")
plt.ylabel("$\Psi$")
plt.legend()
plt.grid()
plt.show()

# Plot reconstructed data
# %% Surface map
fig = plt.figure(figsize=(18, 12), dpi=150)
fig.suptitle("Surface Plots of Functions", fontsize=16, y=0.98)
ax1 = fig.add_subplot(221, projection="3d")
surf1 = ax1.plot_surface(
    xgrid, tgrid, X1.real, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.8
)
ax1.set_title("F1")
ax1.set_xlabel("Space")
ax1.set_ylabel("Time")
ax1.set_zlabel("F1")
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

ax2 = fig.add_subplot(222, projection="3d")
surf2 = ax2.plot_surface(
    xgrid, tgrid, X2.real, cmap=cm.plasma, linewidth=0, antialiased=True, alpha=0.8
)
ax2.set_title("F2")
ax2.set_xlabel("Space")
ax2.set_ylabel("Time")
ax2.set_zlabel("F2")
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

ax3 = fig.add_subplot(223, projection="3d")
surf3 = ax3.plot_surface(
    xgrid, tgrid, X.real, cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.8
)
ax3.set_xlabel("Space")
ax3.set_ylabel("Time")
ax3.set_zlabel("F")
fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)

ax4 = fig.add_subplot(224, projection="3d")
surf4 = ax4.plot_surface(
    xgrid,
    tgrid,
    X_dmd.real.T,
    cmap=cm.coolwarm,
    linewidth=0,
    antialiased=True,
    alpha=0.8,
)
ax4.set_xlabel("Space")
ax4.set_ylabel("Time")
ax4.set_zlabel("F_(recon)")
fig.colorbar(surf4, ax=ax4, shrink=0.5, aspect=10)

plt.tight_layout(pad=2.0)
plt.show()

# %% Compute reconstruction error
error = np.linalg.norm(X - X_dmd.T) / np.linalg.norm(X)
print(f"Relative reconstruction error: {error:.6f}")

# %%
