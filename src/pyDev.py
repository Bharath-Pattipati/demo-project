# %% import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
plt.show()
