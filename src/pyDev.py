# %% import libraries
import numpy as np

# import pandas as pd
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
