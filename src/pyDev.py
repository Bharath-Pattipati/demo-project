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
x = 3  # True slope
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
xtilde2 = np.linalg.pinv(a) @ b
