"""
Python Dynamic Mode Decomposition
Tutorial 1: Basic DMD, https://github.com/mathLab/PyDMD/blob/master/tutorials/tutorial1/tutorial-1-dmd.ipynb
This example demonstrates the basic usage of the DMD class in pydmd.
It shows how to perform DMD on a simple dataset, visualize the results,
and interpret the output.
The example uses a synthetic dataset consisting of two functions
that are summed together.
"""

# %% Import the necessary libraries

import numpy as np
import matplotlib.pyplot as plt

from pydmd import DMD, BOPDMD
from pydmd.plotter import plot_eigs, plot_summary
from pydmd.preprocessing import hankel_preprocessing
import warnings

warnings.filterwarnings("ignore")


# %% Create the input data by summing two different functions
def f1(x, t):
    return 1.0 / np.cosh(x + 3) * np.cos(2.3 * t)


def f2(x, t):
    return 2.0 / np.cosh(x) * np.tanh(x) * np.sin(2.8 * t)


nx = 65  # number of grid points along space dimension
nt = 129  # number of grid points along time dimension

# Define the space and time grid for data collection.
x = np.linspace(-5, 5, nx)
t = np.linspace(0, 4 * np.pi, nt)
xgrid, tgrid = np.meshgrid(x, t)
dt = t[1] - t[0]  # time step between each snapshot

# Data consists of 2 spatiotemporal signals.
X1 = f1(xgrid, tgrid)
X2 = f2(xgrid, tgrid)
X = X1 + X2

# Make a version of the data with noise.
mean = 0
std_dev = 0.2
random_matrix = np.random.normal(mean, std_dev, size=(nt, nx))
Xn = X + random_matrix

X.shape

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

# %% Functions and the dataset with noise.
titles = ["$f_1(x,t)$", "$f_2(x,t)$", "$f$ (noisy)"]
data = [X1, X2, Xn]

fig = plt.figure(figsize=(17, 6), dpi=200)
for n, title, d in zip(range(131, 134), titles, data):
    plt.subplot(n)
    plt.pcolor(xgrid, tgrid, d.real)
    plt.title(title)
    plt.xlabel("Space")
    plt.ylabel("Time")
    plt.colorbar()
plt.show()

# %% DMD with perfect data (i.e. clean simulation data)
d = 2  # we will use this number of delays throughout the tutorial
dmd = DMD(svd_rank=4)
delay_dmd = hankel_preprocessing(dmd, d=d)
delay_dmd.fit(X.T)
plot_summary(delay_dmd, x=x, t=dt, d=d)
print(
    f"Frequencies (imaginary component): {np.round(np.log(delay_dmd.eigs) / dt, decimals=12)}"
)
plt.title("Reconstructed Data")
plt.imshow(delay_dmd.reconstructed_data.real)
plt.show()
plt.title("Clean Ground Truth Data")
plt.imshow(X.T)
plt.show()

# %% DMD steps for handling real data (i.e. data with noise)
# Step 1: Do a time-delay embedding (d is number of delay embeddings).
# Step 2: Apply BOP-DMD (num_trials is number of statistical bags).
# Step 3: OPTIONAL -- Constrain the eigenvalues (i) left-half plane, (ii) imaginary axis, (iii) complex conjugate pairs.

# Build the Optimized DMD model.
# num_trials=0 gives Optimized DMD, without bagging.
optdmd = BOPDMD(svd_rank=4, num_trials=0)

# Wrap the model with the preprocessing routine.
delay_optdmd = hankel_preprocessing(optdmd, d=d)

# Fit the model to the noisy data.
# Note: BOPDMD models need the data X and the times of data collection t for fitting.
# Hence if we apply time-delay, we must adjust the length of our time vector accordingly.
delay_t = t[: -d + 1]
delay_optdmd.fit(Xn.T, t=delay_t)

# Plot a summary of the DMD results.
plot_summary(delay_optdmd, x=x, d=d)

# Print computed eigenvalues (frequencies are given by imaginary components).
# Also plot the resulting data reconstruction.
print(f"Frequencies (imaginary component): {np.round(delay_optdmd.eigs, decimals=3)}")
plt.title("Reconstructed Data")
plt.imshow(delay_optdmd.reconstructed_data.real)
plt.show()
plt.title("Clean Ground Truth Data")
plt.imshow(X.T)
plt.show()


# %% DMD object attributes
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# Plotting the modes individually...
plt.figure(figsize=(14, 3))
for i, mode in enumerate(delay_optdmd.modes.T):
    # Get the average across delays, since we used time-delay.
    mode = np.average(mode.reshape(d, len(mode) // d), axis=0)
    plt.subplot(1, len(delay_optdmd.modes.T), i + 1)
    plt.plot(mode.real, c=colors[i])
    plt.title(f"Mode {i + 1}")
plt.tight_layout()
plt.show()

# Plotting the dynamics individually...
plt.figure(figsize=(14, 3))
for i, dynamic in enumerate(delay_optdmd.dynamics):
    plt.subplot(1, len(delay_optdmd.dynamics), i + 1)
    plt.plot(delay_t, dynamic.real, c=colors[i])
    plt.title(f"Dynamics {i + 1}")
plt.tight_layout()
plt.show()

# Plot the eigenvalues.
plot_eigs(delay_optdmd, show_axes=True, show_unit_circle=False, figsize=(4, 4))

# Print the amplitudes.
print(f"Computed amplitudes: {np.round(delay_optdmd.amplitudes, decimals=3)}\n")

# %% Step 3: DMD with constrained eigenvalues
# BOPDMD models also have the option to specify the structure of the eigenvalues that they compute. More specifically, users can impose the following constraints, as well as any valid combination of them.

# Stable: constrain eigenvalues to have non-positive real parts.
# Imaginary: constrain eigenvalues to be purely imaginary.
# Conjugate pairs: constrain eigenvalues to always appear with their complex conjugate.

# CONSTRAINTS
# Stable: constrain to the left-half plane (no positive real parts to eigenvalues).
# bopdmd = BOPDMD(eig_constraints={"stable"})

# Imaginary: constrain to imaginary axis (no real parts to eigenvalues).
# bopdmd = BOPDMD(eig_constraints={"imag"})

# Stable + Conjugate: constrain to the left-half plane and as complex conjugates.
# bopdmd = BOPDMD(eig_constraints={"stable", "conjugate_pairs"})

# Imaginary + Conjugate: constrain to imaginary axis and as complex conjugates.
# bopdmd = BOPDMD(eig_constraints={"imag", "conjugate_pairs"})

optdmd = BOPDMD(svd_rank=4, num_trials=0, eig_constraints={"imag", "conjugate_pairs"})
delay_optdmd = hankel_preprocessing(optdmd, d=d)
delay_optdmd.fit(Xn.T, t=delay_t)
plot_summary(delay_optdmd, x=x, d=d)

print(f"Frequencies (imaginary component): {np.round(delay_optdmd.eigs, decimals=3)}")
plt.title("Reconstructed Data")
plt.imshow(delay_optdmd.reconstructed_data.real)
plt.show()
plt.title("Clean Ground Truth Data")
plt.imshow(X.T)
plt.show()

# %% Why we need time-delay?
# without time-delay, we obtain a DMD model that is not able to capture the dynamics of the system.
dmd = DMD(svd_rank=4)
dmd.fit(X.T)
plot_summary(dmd)

print(
    f"Frequencies (imaginary component): {np.round(np.log(dmd.eigs) / dt, decimals=3)}"
)
plt.title("Reconstructed Data")
plt.imshow(dmd.reconstructed_data.real)
plt.show()
plt.title("Clean Ground Truth Data")
plt.imshow(X.T)
plt.show()

# %% Why use BOP-DMD?
# Exact DMD instead of BOP-DMD
# BOP-DMD is a statistical method that uses bagging to improve the robustness of DMD.
# BOP-DMD is extremely robust to measurement noise, which is why it the preferred method when dealing with real-world data.
dmd = DMD(svd_rank=4)
delay_dmd = hankel_preprocessing(dmd, d=d)
delay_dmd.fit(Xn.T)
plot_summary(delay_dmd, x=x, d=d)

print(
    f"Frequencies (imaginary component): {np.round(np.log(delay_dmd.eigs) / dt, decimals=3)}"
)
plt.title("Reconstructed Data")
plt.imshow(delay_dmd.reconstructed_data.real)
plt.show()
plt.title("Clean Ground Truth Data")
plt.imshow(X.T)
plt.show()

# %%
